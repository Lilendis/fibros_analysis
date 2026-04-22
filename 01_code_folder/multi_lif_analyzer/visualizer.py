from multi_lif_analyzer_legacy import MultiSampleLifAnalyzer as LegacyMultiSampleLifAnalyzer


class VisualizerMixin:
    create_composite_image = LegacyMultiSampleLifAnalyzer.create_composite_image
    save_original_channels = LegacyMultiSampleLifAnalyzer.save_original_channels
    save_processed_channels = LegacyMultiSampleLifAnalyzer.save_processed_channels
    save_annotated_images = LegacyMultiSampleLifAnalyzer.save_annotated_images
    draw_vesicles_on_channel = LegacyMultiSampleLifAnalyzer.draw_vesicles_on_channel
    draw_vesicles_on_composite = LegacyMultiSampleLifAnalyzer.draw_vesicles_on_composite
