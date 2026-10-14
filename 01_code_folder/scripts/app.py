"""
Streamlit application for interactive vesicle analysis.
Loads pre-converted TIFF files from the preprocessed/ folder.
"""

import streamlit as st
import numpy as np
import json
import tifffile
from pathlib import Path
from vesicle_processor import VesicleProcessor, MacrophageProcessor
import os
from datetime import datetime


# ============================================================================
# STREAMLIT CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="🔬 Vesicle Analysis",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stSidebar { width: 450px; }
    h2 { margin-top: 1.5rem; }
    h3 { margin-top: 1rem; }
    .stats-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
if 'processor' not in st.session_state:
    st.session_state.processor = VesicleProcessor()

if 'preprocessed_dir' not in st.session_state:
    st.session_state.preprocessed_dir = Path(__file__).parent / "preprocessed"

if 'current_data' not in st.session_state:
    st.session_state.current_data = None

if 'params' not in st.session_state:
    st.session_state.params = {
        'min_intensity': 5,
        'collagen_diff_threshold': 0,
        'min_vesicle_size': 2,
        'max_vesicle_size': 200,
        'max_iterations': 15,
        'cluster_threshold_factor': 1.5,
        'igg_percent': 90,
        'collagen_factor': 0.5,
        'protein_brightness': 1.7,
        'nuclei_brightness': 1.0,
        'vesicles_brightness': 1.0,
        'protein_threshold': 50,
        'collagen_percentile': 50,
        'nuclei_percentile': 50,
        'analyze_macrophages': True,
        'expansion_factor': 1.58,
        'nuclei_contrast': 1.0
    }

if 'stats' not in st.session_state:
    st.session_state.stats = None

if 'recalculate_segmentation' not in st.session_state:
    st.session_state.recalculate_segmentation = False

if 'save_button' not in st.session_state:
    st.session_state.save_button = False

if 'export_button' not in st.session_state:
    st.session_state.export_button = False

if 'processed_cache' not in st.session_state:
    st.session_state.processed_cache = {}

if 'selected_protein' not in st.session_state:
    st.session_state.selected_protein = None


# ============================================================================
# CACHING FUNCTIONS
# ============================================================================

@st.cache_data
def load_sample_data(preprocessed_path: str, protein_name: str, group_name: str, sample_name: str):
    """Loads TIFF and metadata with caching."""
    processor = VesicleProcessor()
    data = processor.load_sample(Path(preprocessed_path), protein_name, group_name, sample_name)
    return data


@st.cache_data
def get_available_samples(preprocessed_path: str):
    """Returns available proteins, groups and samples.

    Structure:
        {protein_name: {group_name: [{'dir_name': ..., 'display_name': ...}, ...]}}
    """
    preprocessed = Path(preprocessed_path)
    proteins = {}

    if not preprocessed.exists():
        return {}

    for protein_dir in sorted(preprocessed.iterdir()):
        if not protein_dir.is_dir() or protein_dir.name.startswith('.'):
            continue

        groups = {}

        for group_dir in sorted(protein_dir.iterdir()):
            if not group_dir.is_dir() or group_dir.name.startswith('.'):
                continue

            samples = []

            for sample_dir in sorted(group_dir.iterdir()):
                if not sample_dir.is_dir() or not (sample_dir / "channels.tif").exists():
                    continue

                metadata_path = sample_dir / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        sample_display_name = metadata.get('sample_name', sample_dir.name).strip()
                else:
                    sample_display_name = sample_dir.name

                samples.append({
                    'dir_name': sample_dir.name,
                    'display_name': sample_display_name
                })

            if samples:
                groups[group_dir.name] = samples

        if groups:
            proteins[protein_dir.name] = groups

    return proteins


# ============================================================================
# BATCH ANALYSIS (button in sidebar)
# ============================================================================

def run_batch_analysis(protein_name: str):
    """Runs batch processing for a selected protein using parameters from saved_results."""
    base_preprocessed = Path(__file__).parent / "preprocessed"
    base_saved = Path(__file__).parent / "saved_results"
    base_output = base_saved / "auto_analysis" / protein_name

    marker_preprocessed = base_preprocessed / protein_name
    if not marker_preprocessed.exists():
        st.error(f"❌ Folder preprocessed/{protein_name} not found.")
        return

    base_output.mkdir(parents=True, exist_ok=True)

    default_params = {
        'min_intensity': 115, 'collagen_diff_threshold': 15,
        'min_vesicle_size': 5, 'max_vesicle_size': 50,
        'max_iterations': 15, 'cluster_threshold_factor': 1.5,
        'igg_percent': 95, 'collagen_factor': 0.7,
        'protein_brightness': 1.7, 'nuclei_brightness': 1.0,
        'vesicles_brightness': 1.0, 'protein_threshold': 50,
        'collagen_percentile': 50, 'nuclei_percentile': 50,
        'nuclei_contrast': 1.0,
        'analyze_macrophages': True, 'expansion_factor': 1.58,
    }

    processor = VesicleProcessor()
    total_samples = 0
    processed_count = 0
    skipped_count = 0

    with st.spinner("Processing..."):
        for group_dir in sorted(marker_preprocessed.iterdir()):
            if not group_dir.is_dir():
                continue
            group_name = group_dir.name

            for sample_dir in sorted(group_dir.iterdir()):
                if not sample_dir.is_dir():
                    continue

                tiff_path = sample_dir / "channels.tif"
                if not tiff_path.exists():
                    continue

                total_samples += 1
                try:
                    mouse_number = sample_dir.name.split('_')[0]
                except IndexError:
                    continue

                params_path = base_saved / protein_name / mouse_number / sample_dir.name / "analysis_params.json"
                output_dir = base_output / group_name / sample_dir.name

                if (output_dir / "statistics.csv").exists():
                    skipped_count += 1
                    continue

                if params_path.exists():
                    try:
                        with open(params_path, 'r') as f:
                            user_params = json.load(f)
                        params = {**default_params, **user_params}
                    except Exception:
                        params = dict(default_params)
                else:
                    params = dict(default_params)

                params['igg_percent'] = 95
                params['analyze_macrophages'] = True
                params['expansion_factor'] = 1.58

                try:
                    channels_array = tifffile.imread(str(tiff_path))
                    if channels_array.ndim == 3 and channels_array.shape[2] == 4:
                        channels_array = np.transpose(channels_array, (2, 0, 1))
                    channels = [channels_array[0], channels_array[1], channels_array[2], channels_array[3]]

                    meta_path = sample_dir / "metadata.json"
                    if not meta_path.exists():
                        continue
                    with open(meta_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                except Exception:
                    continue

                try:
                    from batch_process import process_sample_batch
                    process_sample_batch(channels, params, metadata, output_dir, processor)
                    processed_count += 1
                except Exception as e:
                    st.warning(f"Failed on {group_name}/{sample_dir.name}: {e}")

    st.success(f"✅ Batch completed. Processed: {processed_count}, skipped (already done): {skipped_count}, total: {total_samples}")


# ============================================================================
# SAMPLE PROCESSING
# ============================================================================

def process_sample(channels, params, analyze_macrophages, expansion_factor):
    """Processes a sample with current parameters."""
    processor = st.session_state.processor

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
        igg_percent=params['igg_percent'],
    )

    # Apply brightness (after background subtraction)
    vesicles_corrected = processor.adjust_brightness(
        vesicles_corrected,
        params['vesicles_brightness']
    )
    protein_corrected = processor.adjust_brightness(
        protein_corrected,
        params['protein_brightness']
    )

    # Nuclei brightness correction
    nuclei = processor.adjust_brightness(
        nuclei,
        params['nuclei_brightness']
    )

    # Segmentation with clustering
    vesicles_binary, vesicles_labeled, num_vesicles = processor.segment_and_cluster_vesicles(
        vesicles_corrected, collagen,
        min_intensity=params['min_intensity'],
        collagen_diff_threshold=params['collagen_diff_threshold'],
        min_size=params['min_vesicle_size'],
        max_size=params['max_vesicle_size'],
        max_iterations=params['max_iterations'],
        cluster_threshold_factor=params['cluster_threshold_factor']
    )

    # Colocalization analysis
    coloc_stats = processor.analyze_colocalization(
        vesicles_labeled, collagen, protein_corrected,
        nuclei_channel=nuclei,
        protein_threshold=params['protein_threshold'],
        collagen_percentile=params['collagen_percentile'],
        nuclei_percentile=params['nuclei_percentile']
    )

    macrophage_labels = None
    macrophage_stats = {}
    if analyze_macrophages:
        macrophage_labels = MacrophageProcessor.segment_macrophages(
            nuclei, protein_corrected,
            nuclei_percentile=params['nuclei_percentile'],
            protein_threshold=params['protein_threshold'],
            expansion_factor=expansion_factor
        )
        macrophage_stats = MacrophageProcessor.analyze_colocalization(
            macrophage_labels,
            vesicles_binary,
            vesicles_labeled
        )
        coloc_stats.update(macrophage_stats)

    return {
        'channels': [nuclei, collagen, vesicles_raw, protein_raw],
        'nuclei': nuclei,
        'collagen': collagen,
        'vesicles_raw': vesicles_raw,
        'protein_raw': protein_raw,
        'vesicles_corrected': vesicles_corrected,
        'protein_corrected': protein_corrected,
        'vesicles_binary': vesicles_binary,
        'vesicles_labeled': vesicles_labeled,
        'num_vesicles': num_vesicles,
        'coloc_stats': coloc_stats,
        'macrophage_labels': macrophage_labels,
        'macrophage_stats': macrophage_stats
    }


def create_visualization_images(processed_data, params):
    """Creates visualizations taking into account brightness and threshold parameters."""
    try:
        processor = st.session_state.processor
        channels = processed_data['channels']
        vesicles_corrected = processed_data['vesicles_corrected']
        protein_corrected = processed_data['protein_corrected']
        segmentation_labeled = processed_data['vesicles_labeled']
        collagen = processed_data['collagen']
        macrophage_labels = processed_data.get('macrophage_labels', None)

        def to_uint8_norm(channel):
            if channel.dtype == np.uint8:
                return channel
            ch = channel.astype(np.float32)
            if ch.max() > 0:
                ch = (ch / ch.max() * 255).astype(np.uint8)
            else:
                ch = ch.astype(np.uint8)
            return ch

        nuclei_8 = to_uint8_norm(processed_data['nuclei'])
        protein_raw_8 = to_uint8_norm(processed_data['protein_raw'])
        vesicles_raw_8 = to_uint8_norm(processed_data['channels'][2])

        nuclei_disp = np.clip(nuclei_8 * params.get('nuclei_brightness', 1.0), 0, 255).astype(np.uint8)
        nuclei_contrast = params.get('nuclei_contrast', 1.0)
        if nuclei_contrast != 1.0:
            gamma = 1.0 / nuclei_contrast
            nuclei_disp = (255 * ((nuclei_disp / 255.0) ** gamma)).astype(np.uint8)

        classification_mask, vesicle_classes = processor.classify_vesicles(
            segmentation_labeled,
            protein_corrected,
            collagen,
            nuclei_channel=processed_data['nuclei'],
            protein_threshold=params.get('protein_threshold', 50),
            collagen_percentile=params.get('collagen_percentile', 50),
            nuclei_percentile=params.get('nuclei_percentile', 50),
            macrophage_labels=macrophage_labels
        )

        if not vesicle_classes or not isinstance(vesicle_classes, dict):
            vesicle_classes = {1: 0}

        composite_original = processor.create_composite_rgb(
            processed_data['nuclei'],
            processed_data['collagen'],
            processed_data['vesicles_corrected'],
            protein=processed_data['protein_corrected']
        )
        composite_original = processor.add_scale_bar(composite_original)

        composite_corrected = processor.create_composite_rgb(
            nuclei_disp,
            processed_data['collagen'],
            processed_data['vesicles_corrected'],
            protein=processed_data['protein_corrected']
        )

        if macrophage_labels is not None and np.max(macrophage_labels) > 0:
            composite_corrected_with_macrophages = MacrophageProcessor.draw_macrophages_on_composite(
                composite_corrected,
                macrophage_labels,
                color=(255, 165, 0),
                thickness=2
            )
        else:
            composite_corrected_with_macrophages = composite_corrected

        composite_corrected = processor.add_scale_bar(composite_corrected)

        vesicles_original_red = processor.create_vesicles_original_red(channels[2])

        vesicles_with_contours = processor.create_vesicles_with_contours(
            vesicles_corrected,
            segmentation_labeled
        )

        collagen_original_green = processor.create_collagen_original_green(channels[1])
        protein_original_green = processor.create_collagen_original_green(channels[3])
        protein_corrected_green = processor.create_collagen_original_green(processed_data['protein_corrected'])

        colored_overlay = processor.create_colored_vesicle_overlay(
            composite_corrected_with_macrophages,
            segmentation_labeled,
            vesicle_classes,
            colors={
                0: (0, 255, 255),      # cyan – regular vesicle
                1: (255, 0, 255),      # magenta – colocalized with marker
                2: (255, 255, 255),    # white – inside a cell
                3: (255, 255, 0)       # yellow – inside a macrophage
            },
            thickness=2,
            outer_color=(0, 0, 0),
            outer_thickness=1
        )
        colored_overlay = processor.add_scale_bar(colored_overlay)

        return {
            'composite_original': composite_original,
            'composite_corrected': composite_corrected,
            'colored_overlay': colored_overlay,
            'vesicles_original_red': vesicles_original_red,
            'vesicles_with_contours': vesicles_with_contours,
            'collagen_original': collagen_original_green,
            'protein_original': protein_original_green,
            'protein_corrected': protein_corrected_green,
        }
    except Exception as e:
        st.error(f"❌ Error while creating visualizations: {e}")
        raise


# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.title("🔬 Vesicle analysis")

st.sidebar.subheader("📁 Sample selection")

available_samples = get_available_samples(str(st.session_state.preprocessed_dir))

if available_samples:
    protein_names = sorted(available_samples.keys())
    selected_protein = st.sidebar.selectbox(
        "Protein",
        protein_names,
        key='selected_protein'
    )

    if selected_protein in available_samples:
        group_names = sorted(available_samples[selected_protein].keys())
        selected_group = st.sidebar.selectbox(
            "Group",
            group_names,
            key='selected_group'
        )

        if selected_group in available_samples[selected_protein]:
            sample_options = available_samples[selected_protein][selected_group]
            sample_names = [s['display_name'] for s in sample_options]
            sample_dir_names = {s['display_name']: s['dir_name'] for s in sample_options}

            selected_sample_display = st.sidebar.selectbox(
                "Sample",
                sample_names,
                key='selected_sample'
            )

            selected_sample_dir = sample_dir_names[selected_sample_display]

            if st.sidebar.button("📥 Download sample", use_container_width=True):
                with st.spinner("Loading TIFF and metadata..."):
                    try:
                        sample_data = load_sample_data(
                            str(st.session_state.preprocessed_dir),
                            selected_protein,
                            selected_group,
                            selected_sample_dir
                        )
                        st.session_state.current_data = sample_data
                        st.session_state.recalculate_segmentation = True
                    except Exception as e:
                        st.error(f"❌ Loading error: {e}")

            # Batch analysis button
            if st.sidebar.button("🤖 Run batch analysis", use_container_width=True):
                run_batch_analysis(selected_protein)
else:
    st.sidebar.warning("⚠️ Folder preprocessed/ not found or empty")


# ============================================================================
# PARAMETERS (if sample loaded)
# ============================================================================

if st.session_state.current_data is not None:
    st.sidebar.divider()

    with st.sidebar.expander("🔬 Segmentation", expanded=True):
        st.session_state.params['min_intensity'] = st.slider(
            "Min. vesicle intensity",
            min_value=0, max_value=255,
            value=st.session_state.params['min_intensity'],
            step=1,
            help="Pixels below this value are not considered vesicles"
        )

        st.session_state.params['collagen_diff_threshold'] = st.slider(
            "Autofluorescence difference threshold",
            min_value=0, max_value=100,
            value=st.session_state.params['collagen_diff_threshold'],
            step=1,
            help="Vesicles must be brighter than autofluorescence by this value"
        )

        col1, col2 = st.columns(2)
        with col1:
            st.session_state.params['min_vesicle_size'] = st.number_input(
                "Min. size (pix.)",
                min_value=1, max_value=100,
                value=st.session_state.params['min_vesicle_size'],
                step=1
            )
        with col2:
            st.session_state.params['max_vesicle_size'] = st.number_input(
                "Max. size (pix.)",
                min_value=10, max_value=1000,
                value=st.session_state.params['max_vesicle_size'],
                step=10
            )

    with st.sidebar.expander("🔨 Clustering"):
        st.session_state.params['max_iterations'] = st.slider(
            "Max iterations",
            min_value=1, max_value=100,
            value=st.session_state.params['max_iterations'],
            step=1
        )

        st.session_state.params['cluster_threshold_factor'] = st.slider(
            "Clustering threshold",
            min_value=0.01, max_value=3.0,
            value=st.session_state.params['cluster_threshold_factor'],
            step=0.01
        )

        if st.button("⚡ Recalculate clustering", use_container_width=True):
            st.session_state.recalculate_segmentation = True

    with st.sidebar.expander("🧹 Background subtraction"):
        st.session_state.params['igg_percent'] = st.slider(
            "IgG percentile",
            min_value=50, max_value=100,
            value=st.session_state.params['igg_percent'],
            step=1,
            help="Percentile used for IgG background estimation"
        )

        st.session_state.params['collagen_factor'] = st.slider(
            "Autofluorescence factor",
            min_value=0.0, max_value=1.0,
            value=st.session_state.params['collagen_factor'],
            step=0.01
        )

    with st.sidebar.expander("💡 Channel brightness", expanded=False):
        st.session_state.params['vesicles_brightness'] = st.slider(
            "Vesicles brightness (corrected)",
            min_value=0.01, max_value=10.0,
            value=st.session_state.params['vesicles_brightness'],
            step=0.01,
            help="Brightness factor for vesicles after background subtraction"
        )

        st.session_state.params['protein_brightness'] = st.slider(
            "Marker brightness (corrected)",
            min_value=0.01, max_value=10.0,
            value=st.session_state.params['protein_brightness'],
            step=0.01,
            help="Brightness factor for marker after IgG subtraction"
        )

        st.session_state.params['nuclei_brightness'] = st.slider(
            "Nuclei brightness (corrected)",
            min_value=0.01, max_value=10.0,
            value=st.session_state.params['nuclei_brightness'],
            step=0.01,
            help="Brightness factor for nuclei after background subtraction"
        )

        st.session_state.params['nuclei_contrast'] = st.slider(
            "Nuclei contrast",
            min_value=0.5, max_value=5.0,
            value=st.session_state.params.get('nuclei_contrast', 1.0),
            step=0.05,
            help="<1 increases contrast, >1 decreases contrast"
        )

    with st.sidebar.expander("🎯 Colocalization"):
        st.session_state.params['protein_threshold'] = st.slider(
            "Marker threshold",
            min_value=0, max_value=65500,
            value=st.session_state.params['protein_threshold'],
            step=1,
            help="Minimum marker intensity inside a vesicle"
        )

        st.session_state.params['collagen_percentile'] = st.slider(
            "Autofluorescence percentile threshold",
            min_value=0, max_value=100,
            value=st.session_state.params['collagen_percentile'],
            step=1,
            help="Percentile of autofluorescence used as threshold for 'in cell'"
        )

        st.session_state.params['nuclei_percentile'] = st.slider(
            "Nuclei percentile threshold",
            min_value=0, max_value=100,
            value=st.session_state.params['nuclei_percentile'],
            step=1,
            help="Percentile of nuclei used as threshold for 'in cell' (0 = any non-zero, 100 = only brightest)"
        )

    with st.sidebar.expander("🧫 Macrophages"):
        st.session_state.params['analyze_macrophages'] = st.checkbox(
            "Analyze macrophages",
            value=st.session_state.params['analyze_macrophages']
        )
        st.session_state.params['expansion_factor'] = st.slider(
            "Macrophage expansion factor",
            min_value=1.0, max_value=2.5, step=0.05,
            value=st.session_state.params['expansion_factor'],
            help="Factor by which the nucleus radius is increased to cover the macrophage"
        )

    st.sidebar.divider()

    st.sidebar.subheader("📊 Statistics")

    if st.session_state.stats is not None:
        stats = st.session_state.stats

        metrics = [
            ("Total vesicles", stats['coloc_stats']['total_vesicles']),
            ("In cells (nuclei+AF)", stats['coloc_stats']['vesicles_in_cells']),
            ("With marker", stats['coloc_stats']['protein_colocalized']),
            ("Colocalization %", f"{stats['coloc_stats']['protein_colocalization_percent']:.1f}%"),
            ("In cells %", f"{stats['coloc_stats']['percent_in_cells']:.1f}%"),
            ("Col/Cell %", f"{stats['coloc_stats']['colocalization_to_cell_ratio']:.1f}%"),
            ("Mean marker intensity", f"{stats['coloc_stats']['mean_protein_intensity']:.0f}"),
            ("Mean autofluorescence intensity", f"{stats['coloc_stats']['mean_collagen_intensity']:.0f}"),
        ]

        for label, value in metrics:
            st.sidebar.metric(label, value)

        if st.session_state.params.get('analyze_macrophages', False):
            st.sidebar.divider()
            st.sidebar.markdown("**🧫 Macrophages**")
            st.sidebar.metric("Total", stats['coloc_stats'].get('total_macrophages', 0))
            st.sidebar.metric("With vesicles", stats['coloc_stats'].get('macrophages_with_vesicles', 0))
            st.sidebar.metric("With vesicles %", f"{stats['coloc_stats'].get('macrophage_colocalization_percent', 0.0):.1f}%")
            st.sidebar.metric("Vesicles in macrophages", stats['coloc_stats'].get('vesicles_in_macrophages', 0))
            st.sidebar.metric("Mean vesicles per macrophage", f"{stats['coloc_stats'].get('mean_vesicles_per_macrophage', 0.0):.2f}")

    st.sidebar.divider()

    st.session_state.save_button = st.sidebar.button("💾 Save", use_container_width=True)
    st.session_state.export_button = st.sidebar.button("📥 Export CSV", use_container_width=True)

    if st.button("🔄 Reset parameters", use_container_width=True):
        st.session_state.params = {
            'min_intensity': 5,
            'collagen_diff_threshold': 0,
            'min_vesicle_size': 2,
            'max_vesicle_size': 200,
            'max_iterations': 15,
            'cluster_threshold_factor': 1.5,
            'igg_percent': 90,
            'collagen_factor': 0.7,
            'vesicles_brightness': 1.0,
            'protein_brightness': 1.7,
            'nuclei_brightness': 1.0,
            'protein_threshold': 50,
            'collagen_percentile': 50,
            'nuclei_percentile': 50,
            'analyze_macrophages': True,
            'expansion_factor': 1.58,
            'nuclei_contrast': 1.0,
        }
        st.session_state.recalculate_segmentation = True
        st.rerun()


# ============================================================================
# MAIN PANEL
# ============================================================================

if st.session_state.current_data is not None:
    st.title("🔬 Vesicle analysis")

    metadata = st.session_state.current_data.get('metadata', {})
    protein_name = st.session_state.selected_protein or metadata.get('protein', 'N/A')
    group_name = metadata.get('group', 'N/A')
    sample_name = metadata.get('sample_name', 'N/A').strip()

    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        st.subheader(f"📍 {sample_name}")
    with col2:
        st.caption(f"Protein: {protein_name}")
    with col3:
        st.caption(f"Group: {group_name}")
    with col4:
        st.caption(f"Size: {st.session_state.current_data['shape'][0]}×{st.session_state.current_data['shape'][1]}")

    sample_cache_key = json.dumps({
        'protein': protein_name,
        'group': group_name,
        'sample': sample_name,
        'shape': st.session_state.current_data['shape'],
        'params': st.session_state.params
    }, sort_keys=True, default=str)

    cached_result = st.session_state.processed_cache.get(sample_cache_key)

    if cached_result is not None and not st.session_state.recalculate_segmentation:
        processed = cached_result['processed']
        visuals = cached_result['visuals']
    else:
        with st.spinner("🔄 Processing sample..."):
            processed = process_sample(
                st.session_state.current_data['channels'],
                st.session_state.params,
                analyze_macrophages=st.session_state.params['analyze_macrophages'],
                expansion_factor=st.session_state.params['expansion_factor']
            )

            st.session_state.stats = processed
            visuals = create_visualization_images(processed, st.session_state.params)

        st.session_state.processed_cache[sample_cache_key] = {
            'processed': processed,
            'visuals': visuals
        }
        st.session_state.recalculate_segmentation = False

    st.session_state.stats = processed

    if st.session_state.get('save_button', False):
        with st.spinner("💾 Saving results..."):
            try:
                save_dir = Path(__file__).parent / "saved_results"
                output_dir = st.session_state.processor.save_analysis_state(
                    save_dir,
                    st.session_state.current_data['metadata'],
                    st.session_state.params,
                    st.session_state.stats['coloc_stats'],
                    st.session_state.stats,
                    {
                        'composite_original': visuals['composite_original'],
                        'composite_corrected': visuals['composite_corrected'],
                        'colored_overlay': visuals['colored_overlay'],
                        'vesicles_with_contours': visuals['vesicles_with_contours'],
                    }
                )
                st.success(f"✅ Results saved to:\n{output_dir}")
                st.session_state.save_button = False
            except Exception as e:
                st.error(f"❌ Save error: {e}")

    if st.session_state.get('export_button', False):
        with st.spinner("📥 Exporting statistics..."):
            try:
                export_dir = Path(__file__).parent / "exported_stats"
                export_dir.mkdir(exist_ok=True)

                sample_name_slug = st.session_state.current_data['metadata'].get('sample_name', 'sample')
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                csv_file = export_dir / f"stats_{sample_name_slug}_{timestamp}.csv"

                st.session_state.processor.export_statistics_csv(
                    st.session_state.stats['coloc_stats'],
                    csv_file,
                    st.session_state.current_data['metadata']
                )

                st.success(f"✅ Statistics exported to:\n{csv_file}")
                st.session_state.export_button = False
            except Exception as e:
                st.error(f"❌ Export error: {e}")

    st.subheader("📊 Analysis results")

    st.subheader("🔍 Additional channels")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 🟢 Autofluorescence")
        st.image(visuals['collagen_original'])

    with col2:
        st.markdown("#### 🔵 Marker (original)")
        st.image(visuals['protein_original'])

    with col3:
        st.markdown("#### 🟣 Marker (corrected)")
        st.image(visuals['protein_corrected'])

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 1️⃣ Original composite")
        st.markdown("Nuclei | Autofluorescence | Vesicles")
        st.image(visuals['composite_original'])

    with col2:
        st.markdown("#### 2️⃣ Adjusted composite")
        st.markdown("After background subtraction")
        st.image(visuals['composite_corrected'])

    with col3:
        st.markdown("#### 3️⃣ Vesicle segmentation")
        st.markdown(f"Vesicles found: **{processed['num_vesicles']}**")
        st.image(visuals['colored_overlay'])
        st.markdown("""
            **Vesicle colors:**
            - 🔵 **Cyan** – regular vesicle (no colocalization)
            - 🟣 **Magenta** – vesicle colocalized with marker
            - ⚪ **White** – vesicle inside a cell
            - 🟡 **Yellow** – vesicle inside a macrophage
            
            *Each circle has a black outline for better visibility*
            """)

    col4, col5, col6 = st.columns(3)

    with col4:
        st.markdown("#### 4️⃣ Original vesicles")
        st.image(visuals['vesicles_original_red'])

    with col5:
        st.markdown("#### 5️⃣ Vesicles with contours")
        st.markdown("Red — vesicles | Green — contours")
        st.image(visuals['vesicles_with_contours'])

    with col6:
        st.markdown("#### ℹ️ Statistics")
        stats = processed['coloc_stats']
        st.metric("🟢 Total vesicles", stats['total_vesicles'])
        st.metric("In cells", stats['vesicles_in_cells'])
        st.metric("🔵 With marker", stats['protein_colocalized'])
        st.metric("📊 In cells, %", f"{stats['percent_in_cells']:.1f}%")
        st.metric("📊 Colocalization %", f"{stats['protein_colocalization_percent']:.1f}%")
        if st.session_state.params.get('analyze_macrophages', False):
            st.divider()
            st.write("🧫 **Macrophages**")
            st.metric("Total", stats.get('total_macrophages', 0))
            st.metric("With vesicles", stats.get('macrophages_with_vesicles', 0))
            st.metric("% with vesicles", f"{stats.get('macrophage_colocalization_percent', 0.0):.1f}%")
            st.metric("Vesicles inside", stats.get('vesicles_in_macrophages', 0))
            st.metric("Mean vesicles/macrophage", f"{stats.get('mean_vesicles_per_macrophage', 0.0):.2f}")

    st.divider()
    st.subheader("📈 Detailed statistics")

    col1, col2, col3 = st.columns(3)

    with col1:
        stats = processed['coloc_stats']
        st.metric("🟢 Total vesicles", stats['total_vesicles'])
        st.metric("🟡 In cells", stats['vesicles_in_cells'])

    with col2:
        st.metric("🔵 With marker", stats['protein_colocalized'])
        st.metric("📊 Colocalization %", f"{stats['protein_colocalization_percent']:.1f}%")
        st.metric("📊 Col/Cell %", f"{stats['colocalization_to_cell_ratio']:.1f}%")
        st.metric("📊 In cells %", f"{stats['percent_in_cells']:.1f}%")

    with col3:
        st.metric("💾 Mean marker intensity", f"{stats['mean_protein_intensity']:.0f}")
        st.metric("💾 Mean autofluorescence intensity", f"{stats['mean_collagen_intensity']:.0f}")

    if st.session_state.params.get('analyze_macrophages', False):
        st.divider()
        st.subheader("🧫 Detailed macrophage statistics")
        col4, col5, col6 = st.columns(3)
        with col4:
            stats = processed['coloc_stats']
            st.metric("Total macrophages", stats.get('total_macrophages', 0))
            st.metric("Macrophages with vesicles", stats.get('macrophages_with_vesicles', 0))
        with col5:
            st.metric("% macrophages with vesicles", f"{stats.get('macrophage_colocalization_percent', 0.0):.1f}%")
            st.metric("Total vesicles in macrophages", stats.get('vesicles_in_macrophages', 0))
        with col6:
            st.metric("Mean vesicles per macrophage", f"{stats.get('mean_vesicles_per_macrophage', 0.0):.2f}")

    if 'channels' in metadata:
        st.divider()
        st.subheader("ℹ️ Channel information")

        channels_info = metadata['channels']
        col1, col2, col3, col4 = st.columns(4)

        channel_names = ['Nuclei', 'Autofluorescence', 'Vesicles', 'Marker']
        columns = [col1, col2, col3, col4]

        for idx, (col, name) in enumerate(zip(columns, channel_names)):
            with col:
                if str(idx) in channels_info:
                    ch_info = channels_info[str(idx)]
                    st.write(f"**{name}**")
                    st.caption(f"Min: {ch_info.get('min', 'N/A')}")
                    st.caption(f"Max: {ch_info.get('max', 'N/A')}")
                    st.caption(f"Mean: {ch_info.get('mean', 'N/A'):.1f}")

else:
    st.title("🔬 Vesicle analysis - Streamlit")

    st.markdown("""
    ### 👋 Welcome!
    
    This is an interactive application for vesicle analysis in microscopy images.
    
    **Getting started:**
    1. Select a protein, group, and sample in the sidebar
    2. Click "Download sample"
    3. Adjust analysis parameters
    4. View results in real time
    
    **Features:**
    - 🔬 Vesicle segmentation with adaptive parameters
    - 🔨 Cluster splitting via watershed
    - 🎯 Colocalization analysis with marker and autofluorescence
    - 🧫 Macrophage detection and colocalization
    - 📊 Interactive visualizations
    - 💾 Saving results
    
    **Start from the sidebar** 👈
    """)

    if not get_available_samples(str(st.session_state.preprocessed_dir)):
        st.error("""
        ❌ **Error:** The `preprocessed/` folder was not found or is empty.
        
        Make sure:
        - TIFF files follow the structure: `preprocessed/protein/group/sample/channels.tif`
        - Metadata files follow the structure: `preprocessed/protein/group/sample/metadata.json`
        """)
