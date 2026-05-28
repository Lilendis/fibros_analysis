import cv2
import numpy as np
from skimage import exposure, filters

from multi_lif_analyzer_legacy import MultiSampleLifAnalyzer as LegacyMultiSampleLifAnalyzer


class ImageProcessorMixin:
    def preprocess_channel(self, img, channel_type):
        if img is None:
            return None

        try:
            img_denoised = cv2.medianBlur(img, 3)

            try:
                threshold = filters.threshold_otsu(img_denoised)
                background_reduced = cv2.subtract(img_denoised, int(threshold * 0.7))
                background_reduced = np.clip(background_reduced, 0, 255)
            except Exception:
                background_reduced = img_denoised

            if channel_type == 2:
                img_contrast = background_reduced
            else:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                img_contrast = clahe.apply(background_reduced)

            gamma = self.gamma_values.get(channel_type, 0.8)
            img_enhanced = exposure.adjust_gamma(img_contrast, gamma=gamma)

            return img_enhanced

        except Exception as e:
            print(f"   Error preprocessing channel {channel_type}: {e}")
            return None

    def enhance_vesicles_brightness(self, vesicles_channel, enhancement_factor=1.0):
        if vesicles_channel is None:
            return vesicles_channel

        try:
            if enhancement_factor > 1.0:
                vesicles_enhanced = cv2.medianBlur(vesicles_channel, 3)
                vesicles_enhanced = cv2.convertScaleAbs(vesicles_enhanced, alpha=1.1, beta=5)
                print("      Enhanced vesicle brightness")
            else:
                vesicles_enhanced = vesicles_channel

            return vesicles_enhanced

        except Exception as e:
            print(f"      Error in enhance_vesicles_brightness: {e}")
            return vesicles_channel

    enhance_contrast = LegacyMultiSampleLifAnalyzer.enhance_contrast
    aggressive_vesicles_enhancement = LegacyMultiSampleLifAnalyzer.aggressive_vesicles_enhancement
    add_scale_bar = LegacyMultiSampleLifAnalyzer.add_scale_bar
    apply_colored_channel = LegacyMultiSampleLifAnalyzer.apply_colored_channel
    apply_colored_channel_simple = LegacyMultiSampleLifAnalyzer.apply_colored_channel_simple
    apply_black_background = LegacyMultiSampleLifAnalyzer.apply_black_background
