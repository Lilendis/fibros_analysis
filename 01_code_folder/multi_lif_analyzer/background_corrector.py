import cv2
import numpy as np

from multi_lif_analyzer_legacy import MultiSampleLifAnalyzer as LegacyMultiSampleLifAnalyzer


class BackgroundCorrectorMixin:
    def get_vesicle_preserve_mask(self, vesicles_binary, vesicles_original):
        """
        Pixels that keep original red-channel (channel 2) intensity after background subtraction.
        Uses the segmentation mask plus a small dilation and bright-red pixels.
        """
        if vesicles_binary is None or vesicles_original is None:
            return None

        mask = np.asarray(vesicles_binary) > 0
        if not np.any(mask):
            return None

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=2).astype(bool)

        if self.min_vesicle_intensity > 0:
            bright_red = np.asarray(vesicles_original) >= self.min_vesicle_intensity
            mask = mask | bright_red
        else:
            bright_red = np.asarray(vesicles_original) > 0
            mask = mask | bright_red

        return mask

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
        """
        Subtracts the cell-marker background only from non-vesicle regions
        while preserving vesicle brightness.
        """
        if vesicles_channel_raw is None or cell_marker_channel is None or vesicles_binary is None:
            return vesicles_channel_raw

        vesicles_binary_mask = vesicles_binary.copy().astype(bool)
        preserve_mask = self.get_vesicle_preserve_mask(vesicles_binary, vesicles_channel_raw)
        if preserve_mask is None:
            preserve_mask = vesicles_binary_mask
        protected_mask = preserve_mask
        print(f"      Preserve mask pixels (segmentation + halo): {np.sum(protected_mask)}")
        print(f"      Strict segmentation mask pixels: {np.sum(vesicles_binary_mask)}")

        background_mask = ~protected_mask
        print(f"      Total protected pixels: {np.sum(protected_mask)}")
        print(f"      Background pixels for subtraction: {np.sum(background_mask)}")

        result = vesicles_channel_raw.copy().astype(np.float32)

        mean_before_binary = 0.0
        mean_before_protected = 0.0
        mean_before_background = 0.0
        if np.sum(vesicles_binary_mask) > 0:
            mean_before_binary = float(np.mean(result[vesicles_binary_mask]))
            print(f"      Mean intensity INSIDE binary mask BEFORE: {mean_before_binary:.2f}")

        if np.sum(protected_mask) > 0:
            mean_before_protected = float(np.mean(result[protected_mask]))
            print(f"      Mean protected brightness BEFORE: {mean_before_protected:.2f}")

        if np.sum(background_mask) > 0:
            mean_before_background = float(np.mean(result[background_mask]))
            print(f"      Mean background brightness BEFORE: {mean_before_background:.2f}")

        if np.sum(background_mask) > 0:
            safe_factor = min(self.subtraction_factor, 0.4)
            print(f"      Subtraction factor: {self.subtraction_factor} -> {safe_factor}")
            result[background_mask] = np.maximum(
                0,
                result[background_mask] - cell_marker_channel[background_mask].astype(np.float32) * safe_factor,
            )

        result = np.clip(result, 0, 255).astype(np.uint8)

        if np.sum(preserve_mask) > 0:
            result[preserve_mask] = np.asarray(vesicles_channel_raw)[preserve_mask]
            print(f"      Restored original red-channel intensity for {np.sum(preserve_mask)} pixels")

        if np.sum(vesicles_binary_mask) > 0:
            mean_after_binary = float(np.mean(result[vesicles_binary_mask]))
            preserved_binary_percent = (mean_after_binary / mean_before_binary * 100) if mean_before_binary > 0 else 0
            print(f"      Mean intensity INSIDE segmentation mask AFTER: {mean_after_binary:.2f}")
            print(f"      Preserved inside segmentation mask: {preserved_binary_percent:.1f}% (target ~100%)")

        if np.sum(protected_mask) > 0:
            mean_after_protected = float(np.mean(result[protected_mask]))
            preserved_percent = (mean_after_protected / mean_before_protected * 100) if mean_before_protected > 0 else 0
            print(f"      Mean protected brightness AFTER: {mean_after_protected:.2f}")
            print(f"      Preserved brightness: {preserved_percent:.1f}%")

            if preserved_percent < 95:
                print("      Warning: more than 5% vesicle brightness was lost")

        if np.sum(background_mask) > 0:
            mean_after_background = float(np.mean(result[background_mask]))
            print(f"      Mean background brightness AFTER: {mean_after_background:.2f}")

        return result

    def finalize_vesicle_display_channel(self, vesicles_original, vesicles_corrected, vesicles_binary):
        """
        Builds the display channel for corrected vesicles (red / channel 2).
        Background stays subtracted; vesicle regions keep original red-channel intensity.
        No CLAHE/gamma/blur on vesicle pixels.
        """
        if vesicles_corrected is None:
            return vesicles_original

        if vesicles_original is None:
            vesicles_original = vesicles_corrected

        vesicles_original = np.asarray(vesicles_original)
        result = np.asarray(vesicles_corrected).copy()

        preserve_mask = self.get_vesicle_preserve_mask(vesicles_binary, vesicles_original)
        if preserve_mask is None or not np.any(preserve_mask):
            processed = self.preprocess_channel(vesicles_corrected, 2)
            return processed if processed is not None else result

        result[preserve_mask] = vesicles_original[preserve_mask]
        print(
            f"      Display channel: original red intensity on {np.sum(preserve_mask)} px "
            f"(min_vesicle_intensity={self.min_vesicle_intensity} used only for segmentation)"
        )

        seg_mask = np.asarray(vesicles_binary) > 0
        if np.any(seg_mask):
            mean_orig = float(np.mean(vesicles_original[seg_mask]))
            mean_final = float(np.mean(result[seg_mask]))
            preserved_pct = (mean_final / mean_orig * 100) if mean_orig > 0 else 100.0
            print(f"      Mean inside segmentation mask: {mean_orig:.1f} -> {mean_final:.1f} ({preserved_pct:.1f}%)")

        brightness_factor = getattr(self, "vesicle_brightness_factor", 1.0)
        if brightness_factor != 1.0:
            background_mask = ~preserve_mask
            boosted = result.astype(np.float32)
            boosted[background_mask] = np.clip(
                boosted[background_mask] * brightness_factor,
                0,
                255,
            )
            result = boosted.astype(np.uint8)
            result[preserve_mask] = vesicles_original[preserve_mask]

        if self.vesicles_contrast_factor != 1.0:
            background_mask = ~preserve_mask
            enhanced = self.enhance_contrast(result.copy(), max(1.0, self.vesicles_contrast_factor))
            if enhanced is not None:
                result = np.asarray(enhanced)
                result[preserve_mask] = vesicles_original[preserve_mask]

        return result
