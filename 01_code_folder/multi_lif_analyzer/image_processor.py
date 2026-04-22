from multi_lif_analyzer_legacy import MultiSampleLifAnalyzer as LegacyMultiSampleLifAnalyzer


class ImageProcessorMixin:
    preprocess_channel = LegacyMultiSampleLifAnalyzer.preprocess_channel
    enhance_contrast = LegacyMultiSampleLifAnalyzer.enhance_contrast
    enhance_vesicles_brightness = LegacyMultiSampleLifAnalyzer.enhance_vesicles_brightness
    aggressive_vesicles_enhancement = LegacyMultiSampleLifAnalyzer.aggressive_vesicles_enhancement
    add_scale_bar = LegacyMultiSampleLifAnalyzer.add_scale_bar
    apply_colored_channel = LegacyMultiSampleLifAnalyzer.apply_colored_channel
    apply_colored_channel_simple = LegacyMultiSampleLifAnalyzer.apply_colored_channel_simple
    apply_black_background = LegacyMultiSampleLifAnalyzer.apply_black_background
