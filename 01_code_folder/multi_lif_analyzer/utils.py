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
