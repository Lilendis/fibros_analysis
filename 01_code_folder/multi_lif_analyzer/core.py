from .segmenter import SegmenterMixin
from .visualizer import VisualizerMixin
from .lif_loader import LifLoaderMixin
from .image_processor import ImageProcessorMixin
from .background_corrector import BackgroundCorrectorMixin
from .colocalization_analyzer import ColocalizationAnalyzerMixin
from .utils import UtilsMixin
from multi_lif_analyzer_legacy import MultiSampleLifAnalyzer as LegacyMultiSampleLifAnalyzer
from datetime import datetime

import numpy as np
import os
import pandas as pd
from skimage import measure


class MultiSampleLifAnalyzer(
    SegmenterMixin,
    VisualizerMixin,
    LifLoaderMixin,
    ImageProcessorMixin,
    BackgroundCorrectorMixin,
    ColocalizationAnalyzerMixin,
    UtilsMixin,
):
    def __init__(
        self,
        min_vesicle_size=5,
        max_vesicle_size=50,
        background_method='original',
        subtraction_factor=0.7,
        vesicles_contrast_factor=1.0,
        protein_contrast_factor=1.5,
        split_large_clusters=True,
        intensity_ratio_threshold=1.5,
        intensity_diff_threshold=20,
        protein_brightness_factor=1.0,
        save_igg_data_path=None,
        protein_subtraction_percent=90,
        collagen_percentile=90,
        exclude_patterns=None,
        channel_background_percentile=70,
        background_threshold_percentile=85,
        igg_fallback_percentile=90,
        default_protein_background=50.0,
        min_vesicle_intensity=115,
        nuclei_gamma=0.8,
        vesicles_gamma=0.7,
        protein_gamma=0.8,
    ):
        self.gamma_values = {
            0: nuclei_gamma,
            1: 1.0,
            2: vesicles_gamma,
            3: protein_gamma,
        }

        self.channel_background_percentile = channel_background_percentile
        self.background_threshold_percentile = background_threshold_percentile
        self.igg_fallback_percentile = igg_fallback_percentile
        self.default_protein_background = default_protein_background

        self.min_vesicle_size = min_vesicle_size
        self.max_vesicle_size = max_vesicle_size
        self.background_method = background_method
        self.subtraction_factor = subtraction_factor
        self.vesicles_contrast_factor = vesicles_contrast_factor
        self.protein_contrast_factor = protein_contrast_factor
        self.split_large_clusters = split_large_clusters
        self.protein_brightness_factor = protein_brightness_factor
        self.intensity_ratio_threshold = intensity_ratio_threshold
        self.intensity_diff_threshold = intensity_diff_threshold
        self.protein_subtraction_percent = protein_subtraction_percent
        self.collagen_percentile = collagen_percentile
        self.min_vesicle_intensity = min_vesicle_intensity
        self.save_igg_data_path = save_igg_data_path

        self.igg_intensity_data = {}
        self.collagen_intensity_data = {}
        self.protein_intensities_from_igg = []

        if exclude_patterns is None:
            exclude_patterns = ['igg', 'pbs', 'control', 'neg']
        self.exclude_patterns = [p.lower() for p in exclude_patterns]

        self.igg_database = self.load_igg_database() if save_igg_data_path else {}

        print(f"      Analyzer settings:")
        print(f"         Contrast vesicles: {vesicles_contrast_factor}")
        print(f"         Contrast protein: {protein_contrast_factor}")
        print(f"         Split clusters: {'ON' if split_large_clusters else 'OFF'}")
        print(f"         Min vesicle intensity: {min_vesicle_intensity}")

    process_lif_file = LegacyMultiSampleLifAnalyzer.process_lif_file

    def process_single_sample(self, sample_data, output_dir, gallery_root_dir, sample_num, total_samples):
        """Processes a single sample using the updated vesicle localization flow."""
        sample_name = sample_data['sample_name']
        mouse_id = sample_data['mouse_id']
        sample_date = sample_data.get('sample_date', datetime.now().strftime("%Y-%m-%d"))

        print(f"\n[{sample_num}/{total_samples}] Sample: {sample_name}")
        print(f"   Group: {mouse_id}")

        sample_output_dir = self.get_sample_output_dir(gallery_root_dir, sample_data)
        os.makedirs(sample_output_dir, exist_ok=True)

        result = self.create_composite_image(sample_data, sample_output_dir)
        if result[0] is None:
            print("   Error: skipping sample because composite creation failed")
            return None

        composite, nuclei_processed, vesicles_binary, protein_processed, composite_filename, vesicles_original = result

        if vesicles_binary is None:
            print("   Error: vesicle mask is missing")
            return None

        channels = sample_data.get('channels', {})
        cell_marker_channel = channels.get(1)
        if cell_marker_channel is None:
            print("   Error: cell marker channel (channel 1) is missing")
            return None

        print("   Applying collagen subtraction with vesicle protection...")
        vesicles_corrected = self.subtract_collagen_preserve_vesicles(
            vesicles_original,
            cell_marker_channel,
            vesicles_binary,
        )

        localization_result = self.analyze_vesicle_localization(
            vesicles_binary,
            vesicles_corrected,
            cell_marker_channel,
            protein_processed,
            sample_name,
        )

        safe_name = "".join(c for c in sample_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        csv_filename = f"{safe_name}_results.csv"
        csv_path = os.path.join(sample_output_dir, csv_filename)

        vesicle_rows = []
        labeled_vesicles = measure.label(vesicles_binary)
        positive_protein = protein_processed[protein_processed > 0]
        protein_threshold = float(np.percentile(positive_protein, 50)) if positive_protein.size > 0 else 50.0
        colocalized_mask = np.logical_and(vesicles_corrected > 10, protein_processed > protein_threshold)

        for region in measure.regionprops(labeled_vesicles):
            region_mask = labeled_vesicles == region.label
            in_cell = bool(np.any(np.logical_and(region_mask, cell_marker_channel > 0)))
            colocalized = bool(np.any(np.logical_and(region_mask, colocalized_mask)))

            vesicle_rows.append({
                'id': int(region.label),
                'area': int(region.area),
                'centroid_x': float(region.centroid[1]),
                'centroid_y': float(region.centroid[0]),
                'in_cell': in_cell,
                'colocalized': colocalized,
            })

        results_df = pd.DataFrame(vesicle_rows)
        results_df.to_csv(csv_path, index=False, encoding='utf-8')

        gallery_rel_dir = os.path.relpath(sample_output_dir, gallery_root_dir).replace("\\", "/")
        gallery_images = []
        for label, filename in [
            ("Composite", f"{safe_name}_composite.png"),
            ("Annotated Composite", f"{safe_name}_composite_annotated.png"),
            ("Vesicles Original", f"{safe_name}_channel2_vesicles_ORIGINAL.png"),
            ("Vesicles Corrected", f"{safe_name}_channel2_vesicles_CORRECTED.png"),
            ("Vesicles Annotated", f"{safe_name}_channel2_vesicles_annotated.png"),
            ("Protein Original", f"{safe_name}_channel3_protein_ORIGINAL.png"),
            ("Protein Corrected", f"{safe_name}_channel3_protein_CORRECTED.png"),
            ("Nuclei Original", f"{safe_name}_channel0_nuclei_ORIGINAL.png"),
            ("Nuclei Corrected", f"{safe_name}_channel0_nuclei_CORRECTED.png"),
            ("Collagen Original", f"{safe_name}_channel1_collagen_ORIGINAL.png"),
        ]:
            full_path = os.path.join(sample_output_dir, filename)
            if os.path.exists(full_path):
                gallery_images.append({
                    'label': label,
                    'path': f"{gallery_rel_dir}/{filename}",
                })

        summary = {
            'sample_name': sample_name,
            'mouse_id': mouse_id,
            'sample_date': sample_date,
            'composite_image': f"{gallery_rel_dir}/{composite_filename}",
            'results_csv': csv_filename,
            'results_csv_path': f"{gallery_rel_dir}/{csv_filename}",
            'total_vesicles': localization_result['total_vesicles'],
            'vesicles_in_cells': localization_result['vesicles_in_cells'],
            'colocalized_vesicles': localization_result['colocalized_vesicles'],
            'percent_in_cells': localization_result['percent_in_cells'],
            'colocalization_percentage': localization_result['percent_colocalized'],
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'gallery_images': gallery_images,
        }

        print("   Results:")
        print(f"      Total vesicles: {summary['total_vesicles']}")
        print(f"      Vesicles in cells: {summary['vesicles_in_cells']}")
        print(f"      Colocalized vesicles: {summary['colocalized_vesicles']}")
        print(f"      Percent in cells: {summary['percent_in_cells']:.2f}%")
        print(f"      Colocalization percentage: {summary['colocalization_percentage']:.2f}%")

        return summary
