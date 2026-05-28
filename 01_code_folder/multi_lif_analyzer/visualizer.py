import os

import cv2
import numpy as np

from multi_lif_analyzer_legacy import MultiSampleLifAnalyzer as LegacyMultiSampleLifAnalyzer


class VisualizerMixin:
    def create_composite_image(self, sample_data, output_dir):
        self._current_segmentation_output_dir = output_dir
        try:
            result = LegacyMultiSampleLifAnalyzer.create_composite_image(self, sample_data, output_dir)
            return self._redraw_vesicles_from_segmentation_mask(result, sample_data, output_dir)
        finally:
            self._current_segmentation_output_dir = None

    def sync_corrected_vesicles_outputs(
        self,
        sample_data,
        output_dir,
        composite,
        composite_filename,
        vesicles_binary,
        vesicles_corrected,
    ):
        if composite is None or vesicles_corrected is None:
            return composite

        sample_name = sample_data.get("sample_name", "sample")
        safe_name = self.make_safe_name(sample_name)
        corrected_output_path = os.path.join(output_dir, f"{safe_name}_channel2_vesicles_CORRECTED.png")

        corrected_colored = np.zeros((*vesicles_corrected.shape, 3), dtype=np.uint8)
        corrected_colored[:, :, 2] = vesicles_corrected
        corrected_with_scale = self.add_scale_bar(corrected_colored, scale_length_pixels=100, scale_text="100 um")
        cv2.imwrite(corrected_output_path, corrected_with_scale)

        updated_result = (
            composite,
            None,
            vesicles_binary,
            None,
            composite_filename,
            None,
            vesicles_corrected,
        )
        redraw_result = self._redraw_vesicles_from_segmentation_mask(
            updated_result,
            sample_data,
            output_dir,
            vesicles_visual=vesicles_corrected,
        )
        if redraw_result and redraw_result[0] is not None:
            return redraw_result[0]
        return composite

    def _redraw_vesicles_from_segmentation_mask(self, result, sample_data, output_dir, vesicles_visual=None):
        if not result or result[0] is None:
            return result

        if len(result) >= 7:
            composite, nuclei_processed, vesicles_binary, protein_processed, composite_filename, vesicles_original, vesicles_visual = result[:7]
        else:
            composite, nuclei_processed, vesicles_binary, protein_processed, composite_filename, vesicles_original = result
            vesicles_visual = None
        if composite is None:
            return result

        composite = composite.copy()
        sample_name = sample_data.get("sample_name", "sample")
        safe_name = self.make_safe_name(sample_name)
        corrected_path = os.path.join(output_dir, f"{safe_name}_channel2_vesicles_CORRECTED.png")

        if vesicles_visual is None and os.path.exists(corrected_path):
            corrected_image = cv2.imread(corrected_path, cv2.IMREAD_COLOR)
            if corrected_image is not None:
                vesicles_visual = corrected_image[:, :, 2]

        if vesicles_visual is None:
            print("         Fallback: unable to load corrected vesicle channel, keeping legacy composite")
            return result

        composite[:, :, 2] = 0
        vesicles_mask = None
        if hasattr(self, "get_vesicle_preserve_mask") and vesicles_original is not None and vesicles_binary is not None:
            vesicles_mask = self.get_vesicle_preserve_mask(vesicles_binary, vesicles_original)
        if vesicles_mask is None and vesicles_binary is not None and np.sum(vesicles_binary) > 0:
            vesicles_mask = vesicles_binary > 0
        if vesicles_mask is not None and np.any(vesicles_mask):
            composite[vesicles_mask, 2] = vesicles_visual[vesicles_mask]
            print(f"         Vesicles on composite: preserve mask ({np.sum(vesicles_mask)} px)")
        else:
            vesicles_mask = vesicles_visual > 10
            composite[vesicles_mask, 2] = vesicles_visual[vesicles_mask]
            print("         Fallback: using threshold >10")

        composite_with_scale = self.add_scale_bar(composite.copy(), scale_length_pixels=100, scale_text="100 um")
        cv2.imwrite(os.path.join(output_dir, composite_filename), composite_with_scale)

        return (
            composite,
            nuclei_processed,
            vesicles_binary,
            protein_processed,
            composite_filename,
            vesicles_original,
            vesicles_visual,
        )

    save_original_channels = LegacyMultiSampleLifAnalyzer.save_original_channels
    save_processed_channels = LegacyMultiSampleLifAnalyzer.save_processed_channels
    save_annotated_images = LegacyMultiSampleLifAnalyzer.save_annotated_images
    draw_vesicles_on_channel = LegacyMultiSampleLifAnalyzer.draw_vesicles_on_channel
    draw_vesicles_on_composite = LegacyMultiSampleLifAnalyzer.draw_vesicles_on_composite
