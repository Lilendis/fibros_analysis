from multi_lif_analyzer_legacy import MultiSampleLifAnalyzer as LegacyMultiSampleLifAnalyzer


class LifLoaderMixin:
    load_all_samples_from_lif = LegacyMultiSampleLifAnalyzer.load_all_samples_from_lif
    force_load_all_channels = LegacyMultiSampleLifAnalyzer.force_load_all_channels
    pil_to_numpy = LegacyMultiSampleLifAnalyzer.pil_to_numpy
    load_igg_from_file = LegacyMultiSampleLifAnalyzer.load_igg_from_file
    find_igg_samples_in_main_file = LegacyMultiSampleLifAnalyzer.find_igg_samples_in_main_file
    save_igg_sample_images = LegacyMultiSampleLifAnalyzer.save_igg_sample_images
