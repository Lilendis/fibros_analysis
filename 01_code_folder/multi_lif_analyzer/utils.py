import os
from datetime import datetime

from multi_lif_analyzer_legacy import MultiSampleLifAnalyzer as LegacyMultiSampleLifAnalyzer


class UtilsMixin:
    make_safe_name = LegacyMultiSampleLifAnalyzer.make_safe_name
    extract_sample_date = LegacyMultiSampleLifAnalyzer.extract_sample_date
    get_sample_output_dir = LegacyMultiSampleLifAnalyzer.get_sample_output_dir
    create_html_gallery = LegacyMultiSampleLifAnalyzer.create_html_gallery
    extract_sample_info = LegacyMultiSampleLifAnalyzer.extract_sample_info
    detect_igg_samples = LegacyMultiSampleLifAnalyzer.detect_igg_samples
    find_background_threshold = LegacyMultiSampleLifAnalyzer.find_background_threshold
    find_collagen_background_percentile = LegacyMultiSampleLifAnalyzer.find_collagen_background_percentile
    calculate_channel_intensity = LegacyMultiSampleLifAnalyzer.calculate_channel_intensity
    calculate_mean_background = LegacyMultiSampleLifAnalyzer.calculate_mean_background
    analyze_vesicle_preservation = LegacyMultiSampleLifAnalyzer.analyze_vesicle_preservation

    def write_segmentation_diagnostics(self, sample_name, reasons, red_regions_count, total_created_vesicles, output_dir):
        """Saves segmentation diagnostics to a text file."""
        try:
            safe_name = "".join(c for c in sample_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            diag_path = os.path.join(output_dir, f"{safe_name}_segmentation_diagnostics.txt")

            with open(diag_path, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write("SEGMENTATION DIAGNOSTICS\n")
                f.write(f"Sample: {sample_name}\n")
                f.write("=" * 60 + "\n\n")

                f.write("SEGMENTATION PARAMETERS:\n")
                f.write(f"  - min_vesicle_size: {self.min_vesicle_size}\n")
                f.write(f"  - max_vesicle_size: {self.max_vesicle_size}\n")
                f.write(f"  - min_vesicle_intensity: {self.min_vesicle_intensity}\n")
                f.write(f"  - cluster_threshold_factor: {self.cluster_threshold_factor}\n")
                f.write(f"  - split_iterations_factor: {self.split_iterations_factor}\n")
                f.write(f"  - max_split_iterations: {self.max_split_iterations}\n")
                f.write(f"  - split_large_clusters: {self.split_large_clusters}\n\n")

                f.write("SEGMENTATION RESULTS:\n")
                f.write(f"  - Total regions: {red_regions_count}\n")
                f.write(f"  - Total created vesicles: {total_created_vesicles}\n\n")

                f.write("REGION STATUS COUNTS:\n")
                f.write(f"  - Too small: {reasons['too_small']}\n")
                f.write(f"  - Kept single: {reasons['kept_single']}\n")
                f.write(f"  - Split disabled: {reasons['split_disabled']}\n")
                f.write(f"  - Split success: {reasons['split_success']}\n")
                f.write(f"  - Split failed: {reasons['split_failed']}\n")
                f.write(f"  - Errors: {reasons.get('errors', 0)}\n\n")

                f.write("REGION DETAILS:\n")
                f.write("-" * 40 + "\n")
                for region_info in reasons.get('details', []):
                    f.write(
                        f"  Region {region_info['id']}: area={region_info['area']}px, "
                        f"intensity={region_info['intensity']:.1f}, status={region_info['status']}\n"
                    )
                    if 'centroid_x' in region_info and 'centroid_y' in region_info:
                        f.write(
                            f"    Coordinates: ({region_info['centroid_x']:.1f}, {region_info['centroid_y']:.1f})\n"
                        )
                    if region_info.get('reason'):
                        f.write(f"    Reason: {region_info['reason']}\n")

                f.write("\n" + "=" * 60 + "\n")
                f.write(f"Analysis time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

            print(f"      Segmentation diagnostics saved: {diag_path}")
            return diag_path
        except Exception as e:
            print(f"      Warning: failed to save segmentation diagnostics: {e}")
            return None
