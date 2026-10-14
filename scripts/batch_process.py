"""
Batch processing script for automated sample analysis.
Reads input TIFFs from preprocessed, parameters from saved_results (by mouse number),
saves results to auto_analysis.
"""

import sys
import json
import csv
from pathlib import Path
import numpy as np
import tifffile
import imageio

from vesicle_processor import VesicleProcessor, MacrophageProcessor


# ============================================================================
# HELPERS
# ============================================================================

def save_statistics_csv(output_dir, stats):
    """Saves statistics to CSV."""
    csv_path = output_dir / "statistics.csv"

    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=stats.keys(), delimiter=',')
        writer.writeheader()
        writer.writerow(stats)

    print(f"   📊 Statistics saved to {csv_path}")


def save_visualizations_png(output_dir, composite_original, colored_overlay):
    """Saves two visualizations as PNG."""
    imageio.imwrite(output_dir / "composite_original.png", composite_original)
    imageio.imwrite(output_dir / "colored_overlay.png", colored_overlay)
    print(f"   🖼️ Visualizations saved to {output_dir}")


def load_analysis_params(params_path):
    """Loads parameters from JSON and fills missing ones with defaults."""
    default_params = {
        'min_intensity': 115,
        'collagen_diff_threshold': 15,
        'min_vesicle_size': 5,
        'max_vesicle_size': 50,
        'max_iterations': 15,
        'cluster_threshold_factor': 1.5,
        'igg_percent': 90,
        'collagen_factor': 0.7,
        'protein_brightness': 1.7,
        'nuclei_brightness': 1.0,
        'vesicles_brightness': 1.0,
        'protein_threshold': 50,
        'collagen_percentile': 50,
        'nuclei_percentile': 50,
        'nuclei_contrast': 1.0,
        'analyze_macrophages': True,
        'expansion_factor': 1.58,
    }
    if params_path.exists():
        with open(params_path, 'r') as f:
            user_params = json.load(f)
        merged = {**default_params, **user_params}
        merged['igg_percent'] = 95
        merged['analyze_macrophages'] = True
        merged['expansion_factor'] = 1.58
        return merged
    else:
        default_params['igg_percent'] = 95
        default_params['analyze_macrophages'] = True
        default_params['expansion_factor'] = 1.58
        return default_params


def process_sample_batch(channels, params, metadata, output_dir, processor):
    """Processes a single sample and saves all results."""
    nuclei = channels[0]
    collagen = channels[1]
    vesicles_raw = channels[2]
    protein_raw = channels[3]

    # Background subtraction
    vesicles_corrected = processor.subtract_collagen_background(
        vesicles_raw, collagen,
        collagen_factor=params['collagen_factor']
    )
    protein_corrected = processor.subtract_igg_from_protein(
        protein_raw,
        igg_percent=params['igg_percent']
    )

    # Brightness
    vesicles_corrected = processor.adjust_brightness(vesicles_corrected, params['vesicles_brightness'])
    protein_corrected = processor.adjust_brightness(protein_corrected, params['protein_brightness'])
    nuclei = processor.adjust_brightness(nuclei, params['nuclei_brightness'])

    # Vesicle segmentation
    vesicles_binary, vesicles_labeled, num_vesicles = processor.segment_and_cluster_vesicles(
        vesicles_corrected, collagen,
        min_intensity=params['min_intensity'],
        collagen_diff_threshold=params['collagen_diff_threshold'],
        min_size=params['min_vesicle_size'],
        max_size=params['max_vesicle_size'],
        max_iterations=params['max_iterations'],
        cluster_threshold_factor=params['cluster_threshold_factor']
    )

    # Colocalization
    coloc_stats = processor.analyze_colocalization(
        vesicles_labeled, collagen, protein_corrected,
        nuclei_channel=nuclei,
        protein_threshold=params['protein_threshold'],
        collagen_percentile=params['collagen_percentile'],
        nuclei_percentile=params['nuclei_percentile']
    )

    # Macrophages (always enabled in batch)
    macrophage_labels = MacrophageProcessor.segment_macrophages(
        nuclei, protein_corrected,
        nuclei_percentile=params['nuclei_percentile'],
        protein_threshold=params['protein_threshold'],
        expansion_factor=params['expansion_factor']
    )
    macrophage_stats = MacrophageProcessor.analyze_colocalization(
        macrophage_labels,
        vesicles_binary,
        vesicles_labeled
    )
    coloc_stats.update(macrophage_stats)

    # Classification (with macrophages)
    classification_mask, vesicle_classes = processor.classify_vesicles(
        vesicles_labeled,
        protein_corrected,
        collagen,
        nuclei_channel=nuclei,
        protein_threshold=params['protein_threshold'],
        collagen_percentile=params['collagen_percentile'],
        nuclei_percentile=params['nuclei_percentile'],
        macrophage_labels=macrophage_labels
    )

    # Visualizations
    def to_uint8(ch):
        if ch.dtype == np.uint8:
            return ch
        ch = ch.astype(np.float32)
        if ch.max() > 0:
            ch = (ch / ch.max() * 255).astype(np.uint8)
        else:
            ch = ch.astype(np.uint8)
        return ch

    nuclei_disp = to_uint8(nuclei)
    vesicles_disp = to_uint8(vesicles_corrected)
    protein_disp = to_uint8(protein_corrected)
    collagen_disp = to_uint8(collagen)

    nuclei_disp = np.clip(nuclei_disp * params['nuclei_brightness'], 0, 255).astype(np.uint8)
    vesicles_disp = np.clip(vesicles_disp * params['vesicles_brightness'], 0, 255).astype(np.uint8)
    protein_disp = np.clip(protein_disp * params['protein_brightness'], 0, 255).astype(np.uint8)

    if 'nuclei_contrast' in params and params['nuclei_contrast'] != 1.0:
        gamma = 1.0 / params['nuclei_contrast']
        nuclei_disp = (255 * ((nuclei_disp / 255.0) ** gamma)).astype(np.uint8)

    # 1. composite_original.png – after subtraction, no macrophages, no vesicles
    composite_original = processor.create_composite_rgb(
        nuclei_disp, collagen_disp, vesicles_disp,
        protein=protein_disp
    )
    composite_original = processor.add_scale_bar(composite_original)

    # 2. colored_overlay.png – after subtraction + macrophages + colored vesicles
    composite_with_macrophages = processor.create_composite_rgb(
        nuclei_disp, collagen_disp, vesicles_disp,
        protein=protein_disp
    )
    if macrophage_labels is not None and np.max(macrophage_labels) > 0:
        composite_with_macrophages = MacrophageProcessor.draw_macrophages_on_composite(
            composite_with_macrophages,
            macrophage_labels,
            color=(255, 165, 0),
            thickness=2
        )
    composite_with_macrophages = processor.add_scale_bar(composite_with_macrophages)

    colored_overlay = processor.create_colored_vesicle_overlay(
        composite_with_macrophages,
        vesicles_labeled,
        vesicle_classes,
        colors={
            0: (0, 255, 255),      # cyan – regular
            1: (255, 0, 255),      # magenta – colocalized with marker
            2: (255, 255, 255),    # white – in cell
            3: (0, 0, 255)         # blue – in macrophage
        },
        thickness=2,
        outer_color=(0, 0, 0),
        outer_thickness=1
    )
    colored_overlay = processor.add_scale_bar(colored_overlay)

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    save_statistics_csv(output_dir, coloc_stats)
    save_visualizations_png(output_dir, composite_original, colored_overlay)

    return output_dir, num_vesicles


# ============================================================================
# MAIN
# ============================================================================

def main():
    if len(sys.argv) != 2:
        print("Usage: python batch_process.py <marker_name>")
        print("Example: python batch_process.py \"IHC CD68 647 EVs 544\"")
        sys.exit(1)

    marker_name = sys.argv[1]

    base_preprocessed = Path(__file__).parent / "preprocessed"
    base_saved = Path(__file__).parent / "saved_results"
    base_output = base_saved / "auto_analysis" / marker_name

    marker_preprocessed = base_preprocessed / marker_name
    if not marker_preprocessed.exists():
        print(f"Error: folder preprocessed/{marker_name} not found.")
        sys.exit(1)

    base_output.mkdir(parents=True, exist_ok=True)

    processor = VesicleProcessor()

    for group_dir in sorted(marker_preprocessed.iterdir()):
        if not group_dir.is_dir():
            continue
        group_name = group_dir.name

        for sample_dir in sorted(group_dir.iterdir()):
            if not sample_dir.is_dir():
                continue

            tiff_path = sample_dir / "channels.tif"
            if not tiff_path.exists():
                print(f"⚠️ Skipping {sample_dir.name}: no channels.tif")
                continue

            try:
                mouse_number = sample_dir.name.split('_')[0]
            except IndexError:
                print(f"⚠️ Skipping {sample_dir.name}: cannot determine mouse number")
                continue

            params_path = base_saved / marker_name / mouse_number / sample_dir.name / "analysis_params.json"
            if not params_path.exists():
                print(f"⚠️ Skipping {sample_dir.name}: no analysis_params.json at {params_path}")
                continue

            output_dir = base_output / group_name / sample_dir.name
            if (output_dir / "statistics.csv").exists():
                print(f"⏭️ Skipping {group_name}/{sample_dir.name}: already processed")
                continue

            try:
                params = load_analysis_params(params_path)
            except Exception as e:
                print(f"❌ Error loading parameters for {sample_dir.name}: {e}")
                sys.exit(1)

            try:
                channels_array = tifffile.imread(str(tiff_path))
                if channels_array.ndim == 3 and channels_array.shape[2] == 4:
                    channels_array = np.transpose(channels_array, (2, 0, 1))
                channels = [channels_array[0], channels_array[1], channels_array[2], channels_array[3]]

                meta_path = sample_dir / "metadata.json"
                if not meta_path.exists():
                    print(f"⚠️ Skipping {sample_dir.name}: no metadata.json")
                    continue
                with open(meta_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            except Exception as e:
                print(f"❌ Error loading data for {sample_dir.name}: {e}")
                sys.exit(1)

            print(f"🔄 Processing: {group_name}/{sample_dir.name} ... ", end="")
            try:
                result_dir, num_ves = process_sample_batch(
                    channels, params, metadata, output_dir, processor
                )
                print(f"✅ vesicles found: {num_ves}")
            except Exception as e:
                print(f"❌ Error: {e}")
                print(f"   Sample: {sample_dir}")
                sys.exit(1)

    print("\n✅ Batch processing completed!")
    print(f"📁 Results saved to: {base_output}")


if __name__ == "__main__":
    main()
