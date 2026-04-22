import cv2
import numpy as np

from multi_lif_analyzer_legacy import MultiSampleLifAnalyzer as LegacyMultiSampleLifAnalyzer


class BackgroundCorrectorMixin:
    subtract_vesicles_background_auto = LegacyMultiSampleLifAnalyzer.subtract_vesicles_background_auto
    simple_background_removal = LegacyMultiSampleLifAnalyzer.simple_background_removal
    subtract_vesicles_background_local = LegacyMultiSampleLifAnalyzer.subtract_vesicles_background_local
    subtract_vesicles_background_interactive = LegacyMultiSampleLifAnalyzer.subtract_vesicles_background_interactive
    load_igg_database = LegacyMultiSampleLifAnalyzer.load_igg_database
    save_igg_database = LegacyMultiSampleLifAnalyzer.save_igg_database
    update_igg_database = LegacyMultiSampleLifAnalyzer.update_igg_database
    get_protein_igg_background = LegacyMultiSampleLifAnalyzer.get_protein_igg_background
    subtract_background_intensity = LegacyMultiSampleLifAnalyzer.subtract_background_intensity
    find_vesicles_by_intensity_difference = LegacyMultiSampleLifAnalyzer.find_vesicles_by_intensity_difference
    _visualize_subtraction = LegacyMultiSampleLifAnalyzer._visualize_subtraction

    def subtract_collagen_preserve_vesicles(
        self,
        vesicles_channel_raw,
        cell_marker_channel,
        vesicles_binary,
    ):
        protected_mask = vesicles_binary
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        protected_mask = cv2.dilate(protected_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
        background_mask = ~protected_mask

        result = vesicles_channel_raw.copy().astype(np.float32)

        protected_pixels = int(np.sum(protected_mask))
        background_pixels = int(np.sum(background_mask))

        if protected_pixels > 0:
            mean_before = float(np.mean(vesicles_channel_raw[protected_mask]))
        else:
            mean_before = 0.0

        result[background_mask] = np.maximum(
            0,
            result[background_mask] - cell_marker_channel[background_mask].astype(np.float32) * self.subtraction_factor,
        )

        result = np.clip(result, 0, 255).astype(np.uint8)

        if protected_pixels > 0:
            mean_after = float(np.mean(result[protected_mask]))
        else:
            mean_after = 0.0

        if mean_before > 0:
            preserved_percent = mean_after / mean_before * 100.0
        else:
            preserved_percent = 0.0

        print(f"      Protected pixels: {protected_pixels}")
        print(f"      Background pixels: {background_pixels}")
        print(f"      Mean vesicle brightness before subtraction: {mean_before:.2f}")
        print(f"      Mean vesicle brightness after subtraction: {mean_after:.2f}")
        print(f"      Preserved brightness: {preserved_percent:.2f}%")

        return result
