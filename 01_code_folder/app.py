"""
Streamlit приложение для интерактивного анализа везикул.
Загружает предварительно сконвертированные TIFF файлы из папки preprocessed/.
"""

from platform import processor
import streamlit as st
import numpy as np
import json
from pathlib import Path
from vesicle_processor import VesicleProcessor, MacrophageProcessor
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
from datetime import datetime


# ============================================================================
# КОНФИГУРАЦИЯ STREAMLIT
# ============================================================================

st.set_page_config(
    page_title="🔬 Vesicle Analysis",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Кастомный CSS для улучшения интерфейса
st.markdown("""
<style>
    /* Левая колонка (узкая) */
    .stSidebar {
        width: 450px;
    }
    
    /* Улучшение читаемости */
    h2 { margin-top: 1.5rem; }
    h3 { margin-top: 1rem; }
    
    /* Статистика */
    .stats-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# ИНИЦИАЛИЗАЦИЯ SESSION STATE
# ============================================================================
if 'processor' not in st.session_state:
    st.session_state.processor = VesicleProcessor()

if 'preprocessed_dir' not in st.session_state:
    st.session_state.preprocessed_dir = Path(__file__).parent / "preprocessed"

if 'current_data' not in st.session_state:
    st.session_state.current_data = None

if 'params' not in st.session_state:
    st.session_state.params = {
        'min_intensity': 5,  # ⭐ Снижено для везикул (макс = 125, средн = 8.5)
        'collagen_diff_threshold': 0,  # ⭐ Отключено (используем только интенсивность)
        'min_vesicle_size': 2,  # ⭐ Снижено (везикулы маленькие)
        'max_vesicle_size': 200,  # ⭐ Увеличено (данные недостаточно большие)
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
# ФУНКЦИИ КЭШИРОВАНИЯ
# ============================================================================

@st.cache_data
def load_sample_data(preprocessed_path: str, protein_name: str, group_name: str, sample_name: str):
    """Загружает TIFF и метаданные с кэшированием."""
    processor = VesicleProcessor()
    data = processor.load_sample(Path(preprocessed_path), protein_name, group_name, sample_name)
    return data


@st.cache_data
def get_available_samples(preprocessed_path: str):
    """Получает доступные белки, группы и образцы.

    Возвращает структуру:
        {protein_name: {group_name: [{'dir_name': ..., 'display_name': ...}, ...]}}
    """
    preprocessed = Path(preprocessed_path)
    proteins = {}

    if not preprocessed.exists():
        return {}

    # Первый уровень — белки
    for protein_dir in sorted(preprocessed.iterdir()):
        if not protein_dir.is_dir() or protein_dir.name.startswith('.'):
            continue

        groups = {}

        # Второй уровень — группы (таймпоинты)
        for group_dir in sorted(protein_dir.iterdir()):
            if not group_dir.is_dir() or group_dir.name.startswith('.'):
                continue

            samples = []

            # Третий уровень — образцы (replicates)
            for sample_dir in sorted(group_dir.iterdir()):
                if not sample_dir.is_dir() or not (sample_dir / "channels.tif").exists():
                    continue

                # Читаем метаданные для красивого названия
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
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ОБРАБОТКИ
# ============================================================================

def process_sample(channels, params, analyze_macrophages, expansion_factor):
    """
    Обрабатывает образец с текущими параметрами.
    """
    processor = st.session_state.processor
    
    nuclei = channels[0]
    collagen = channels[1]
    vesicles_raw = channels[2]
    protein_raw = channels[3]
    
    # Вычитание фона
    vesicles_corrected = processor.subtract_collagen_background(
        vesicles_raw, collagen,
        collagen_factor=params['collagen_factor']
    )
    
    protein_corrected = processor.subtract_igg_from_protein(
        protein_raw,
        igg_percent=params['igg_percent'],
    )

    # ⭐ ПРИМЕНЯЕМ ЯРКОСТЬ (после вычитания фона)
    vesicles_corrected = processor.adjust_brightness(
        vesicles_corrected, 
        params['vesicles_brightness']
    )
    protein_corrected = processor.adjust_brightness(
        protein_corrected, 
        params['protein_brightness']
    )
    
    # Коррекция яркости ядер
    nuclei = processor.adjust_brightness(
        nuclei, 
        params['nuclei_brightness']
    )
    
    # Сегментация с кластеризацией
    vesicles_binary, vesicles_labeled, num_vesicles = processor.segment_and_cluster_vesicles(
        vesicles_corrected, collagen,
        min_intensity=params['min_intensity'],
        collagen_diff_threshold=params['collagen_diff_threshold'],
        min_size=params['min_vesicle_size'],
        max_size=params['max_vesicle_size'],
        max_iterations=params['max_iterations'],
        cluster_threshold_factor=params['cluster_threshold_factor']
    )
    
    # Анализ колокализации
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
            vesicles_binary,  # это сырая маска везикул (до удаления больших объектов)
            vesicles_labeled
        )
        coloc_stats.update(macrophage_stats)

    return {
        'channels': [nuclei, collagen, vesicles_raw, protein_raw],  # ← ДОБАВЛЕНО
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
    """Создает визуализации с учётом параметров яркости и порогов."""
    try:
        processor = st.session_state.processor
        channels = processed_data['channels']
        vesicles_corrected = processed_data['vesicles_corrected']
        protein_corrected = processed_data['protein_corrected']
        segmentation_labeled = processed_data['vesicles_labeled']
        collagen = processed_data['collagen']
        macrophage_labels = processed_data.get('macrophage_labels', None)
        
        # ---------- Нормализация и настройка отображения каналов ----------
        def to_uint8_norm(channel):
            if channel.dtype == np.uint8:
                return channel
            ch = channel.astype(np.float32)
            if ch.max() > 0:
                ch = (ch / ch.max() * 255).astype(np.uint8)
            else:
                ch = ch.astype(np.uint8)
            return ch

        # Базовые 8-битные копии
        nuclei_8 = to_uint8_norm(processed_data['nuclei'])
        vesicles_8 = to_uint8_norm(processed_data['vesicles_corrected'])
        protein_8 = to_uint8_norm(processed_data['protein_corrected'])
        protein_raw_8 = to_uint8_norm(processed_data['protein_raw'])
        vesicles_raw_8 = to_uint8_norm(processed_data['channels'][2])

        # Применяем яркость (если есть)
        nuclei_disp = np.clip(nuclei_8 * params.get('nuclei_brightness', 1.0), 0, 255).astype(np.uint8)
        # Применяем контраст (гамма-коррекция)
        nuclei_contrast = params.get('nuclei_contrast', 1.0)
        if nuclei_contrast != 1.0:
            gamma = 1.0 / nuclei_contrast
            nuclei_disp = (255 * ((nuclei_disp / 255.0) ** gamma)).astype(np.uint8)

        # ---------- Классификация везикул ----------
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

        # Проверяем что vesicle_classes не пустой
        if not vesicle_classes or not isinstance(vesicle_classes, dict):
            vesicle_classes = {1: 0}  # по умолчанию одна везикула класса 0

        # 1. Оригинальный композит
        composite_original = processor.create_composite_rgb(
            processed_data['nuclei'],
            processed_data['collagen'],
            processed_data['vesicles_corrected'],
            protein=processed_data['protein_corrected']
        )
        composite_original = processor.add_scale_bar(composite_original)
        
        # 2. Корректированный композит
        composite_corrected = processor.create_composite_rgb(
            nuclei_disp,
            processed_data['collagen'],
            processed_data['vesicles_corrected'],
            protein=processed_data['protein_corrected']
        )
        if macrophage_labels is not None and np.max(macrophage_labels) > 0:
            composite_corrected_1 = MacrophageProcessor.draw_macrophages_on_composite(
                composite_corrected,
                macrophage_labels,
                color=(255, 165, 0),  # оранжевый
                thickness=2
            )
        composite_corrected = processor.add_scale_bar(composite_corrected)

        
        
        # 4. Оригинальный канал везикул (красный)
        vesicles_original_red = processor.create_vesicles_original_red(channels[2])
        
        # 5. Везикулы с аннотациями (контуры всех везикул)
        vesicles_with_contours = processor.create_vesicles_with_contours(
            vesicles_corrected,
            segmentation_labeled
        )
        
        # 6. Коллаген оригинальный
        collagen_original_green = processor.create_collagen_original_green(channels[1])

        # 7. Белок оригинальный
        protein_original_green = processor.create_collagen_original_green(channels[3])

        # 8. Белок скорректированный
        protein_corrected_green = processor.create_collagen_original_green(processed_data['protein_corrected'])
        
        # 9. Цветной overlay с везикулами
        colored_overlay = processor.create_colored_vesicle_overlay(
            composite_corrected_1,
            segmentation_labeled,
            vesicle_classes,
            colors={
                0: (0, 255, 255),      # желтый обычная
                1: (255, 255, 255),      # белый с белком
                2: (255, 0, 255),     # манджета для везикул с коллагеном
                3: (255, 255, 0)
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
        st.error(f"❌ Ошибка при создании визуализаций: {e}")
        raise


# ============================================================================
# БОКОВАЯ ПАНЕЛЬ (ЛЕВАЯ КОЛОНКА)
# ============================================================================

st.sidebar.title("🔬 Анализ везикул")

# Выбор образца
st.sidebar.subheader("📁 Выбор образца")

available_samples = get_available_samples(str(st.session_state.preprocessed_dir))

if available_samples:
    # Выпадающий список белков
    protein_names = sorted(available_samples.keys())
    selected_protein = st.sidebar.selectbox(
        "Белок",
        protein_names,
        key='selected_protein'
    )

    # Выпадающий список групп для выбранного белка
    if selected_protein in available_samples:
        group_names = sorted(available_samples[selected_protein].keys())
        selected_group = st.sidebar.selectbox(
            "Группа",
            group_names,
            key='selected_group'
        )

        # Выпадающий список образцов в группе
        if selected_group in available_samples[selected_protein]:
            sample_options = available_samples[selected_protein][selected_group]
            sample_names = [s['display_name'] for s in sample_options]
            sample_dir_names = {s['display_name']: s['dir_name'] for s in sample_options}

            selected_sample_display = st.sidebar.selectbox(
                "Образец",
                sample_names,
                key='selected_sample'
            )

            selected_sample_dir = sample_dir_names[selected_sample_display]

            # Кнопка загрузки
            if st.sidebar.button("📥 Загрузить образец", use_container_width=True):
                with st.spinner("Загружаю TIFF и метаданные..."):
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
                        st.error(f"❌ Ошибка загрузки: {e}")
else:
    st.sidebar.warning("⚠️ Папка preprocessed/ не найдена или пуста")

# ============================================================================
# ЭЛЕМЕНТЫ УПРАВЛЕНИЯ (ЕСЛИ ОБРАЗЕЦ ЗАГРУЖЕН)
# ============================================================================

if st.session_state.current_data is not None:
    st.sidebar.divider()
    
    # Параметры сегментации
    with st.sidebar.expander("🔬 Сегментация", expanded=True):
        st.session_state.params['min_intensity'] = st.slider(
            "Мин. интенсивность везикул",
            min_value=0,
            max_value=255,
            value=st.session_state.params['min_intensity'],
            step=1,
            help="Пиксели ниже этого значения не считаются везикулами",
        )
        
        st.session_state.params['collagen_diff_threshold'] = st.slider(
            "Порог разницы с коллагеном",
            min_value=0,
            max_value=100,
            value=st.session_state.params['collagen_diff_threshold'],
            step=1,
            help="Везикулы должны быть ярче коллагена на это значение"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.params['min_vesicle_size'] = st.number_input(
                "Мин. размер (пикс.)",
                min_value=1,
                max_value=100,
                value=st.session_state.params['min_vesicle_size'],
                step=1
            )
        with col2:
            st.session_state.params['max_vesicle_size'] = st.number_input(
                "Макс. размер (пикс.)",
                min_value=10,
                max_value=1000,
                value=st.session_state.params['max_vesicle_size'],
                step=10
            )
    
    # Параметры кластеризации
    with st.sidebar.expander("🔨 Кластеризация"):
        st.session_state.params['max_iterations'] = st.slider(
            "Макс. итераций",
            min_value=1,
            max_value=100,
            value=st.session_state.params['max_iterations'],
            step=1
        )
        
        st.session_state.params['cluster_threshold_factor'] = st.slider(
            "Порог кластеризации",
            min_value=0.01,
            max_value=3.0,
            value=st.session_state.params['cluster_threshold_factor'],
            step=0.01
        )
        
        if st.button("⚡ Пересчитать кластеризацию", use_container_width=True):
            st.session_state.recalculate_segmentation = True
    
    # Параметры вычитания фона
    with st.sidebar.expander("🧹 Вычитание фона"):
        st.session_state.params['igg_percent'] = st.slider(
            "IgG процент",
            min_value=50,
            max_value=100,
            value=st.session_state.params['igg_percent'],
            step=1,
            help="Процентиль для определения IgG фона"
        )
        
        st.session_state.params['collagen_factor'] = st.slider(
            "Коэффициент коллагена",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.params['collagen_factor'],
            step=0.01
        )
        
    with st.sidebar.expander("💡 Яркость каналов", expanded=False):
        st.session_state.params['vesicles_brightness'] = st.slider(
            "Яркость везикул (скорректированных)",
            min_value=0.01,
            max_value=10.0,
            value=st.session_state.params['vesicles_brightness'],
            step=0.01,
            help="Коэффициент яркости для везикул после вычитания фона"
        )
        
        st.session_state.params['protein_brightness'] = st.slider(
            "Яркость белка (скорректированного)",
            min_value=0.01,
            max_value=10.0,
            value=st.session_state.params['protein_brightness'],
            step=0.01,
            help="Коэффициент яркости для белка после вычитания IgG"
        )

        st.session_state.params['nuclei_brightness'] = st.slider(
            "Яркость ядер (скорректированных)",
            min_value=0.01,
            max_value=10.0,
            value=st.session_state.params['nuclei_brightness'],
            step=0.01,
            help="Коэффициент яркости для ядер после вычитания фона"
        )

        st.session_state.params['nuclei_contrast'] = st.slider(
            "Контраст ядер",
            min_value=0.5,
            max_value=5.0,
            value=st.session_state.params.get('nuclei_contrast', 1.0),
            step=0.05,
            help="<1 – повышает контраст (светлее тени), >1 – понижает контраст (темнее)"
        )
    
    # Параметры колокализации
    with st.sidebar.expander("🎯 Колокализация"):
        st.session_state.params['protein_threshold'] = st.slider(
            "Порог белка",
            min_value=0,
            max_value=65500,
            value=st.session_state.params['protein_threshold'],
            step=1,
            help="Минимальная интенсивность белка в везикуле"
        )

        st.session_state.params['collagen_percentile'] = st.slider(
            "Порог коллагена (перцентиль)", 
            min_value=0,
            max_value=100,
            value=st.session_state.params['collagen_percentile'],
            step=1,
            help="Процент пикселей коллагена, используемый как порог для определения 'в клетке'"
        )
        st.session_state.params['nuclei_percentile'] = st.slider(
            "Порог ядер (перцентиль)",
            min_value=0,
            max_value=100,
            value=st.session_state.params['nuclei_percentile'],
            step=1,
            help="Процент пикселей ядер, используемый как порог для определения 'в клетке' (0 = любая ненулевая интенсивность, 100 = только самые яркие)"
        )

    with st.sidebar.expander("🧫 Макрофаги"):
        st.session_state.params['analyze_macrophages'] = st.checkbox(
            "Анализировать макрофаги",
            value=st.session_state.params['analyze_macrophages']
        )
        st.session_state.params['expansion_factor'] = st.slider(
            "Коэффициент расширения макрофага",
            min_value=1.0, max_value=2.5, step=0.05,
            value=st.session_state.params['expansion_factor'],
            help="Во сколько раз увеличивается радиус ядра для определения области макрофага"
        )
    
    st.sidebar.divider()
    
    # Live статистика
    st.sidebar.subheader("📊 Статистика")
    
    if st.session_state.stats is not None:
        stats = st.session_state.stats
        
        # Форматированная статистика
        metrics = [
            ("Везикул всего", stats['coloc_stats']['total_vesicles']),
            ("В клетке (ядра+коллаген)", stats['coloc_stats']['vesicles_in_cells']),
            ("С белком", stats['coloc_stats']['protein_colocalized']),
            ("Колокализация %", f"{stats['coloc_stats']['protein_colocalization_percent']:.1f}%"),
            ("В клетке %", f"{stats['coloc_stats']['percent_in_cells']:.1f}%"),
            ("📊 Кол/Клетка, %", f"{stats['coloc_stats']['colocalization_to_cell_ratio']:.1f}%"),
            ("Средн. интенс. белка", f"{stats['coloc_stats']['mean_protein_intensity']:.0f}"),
            ("Средн. интенс. коллагена", f"{stats['coloc_stats']['mean_collagen_intensity']:.0f}"),
        ]
        
        for label, value in metrics:
            st.sidebar.metric(label, value)

        if st.session_state.params.get('analyze_macrophages', False):
            st.sidebar.divider()
            st.sidebar.markdown("**🧫 Макрофаги**")
            st.sidebar.metric("Всего", stats['coloc_stats'].get('total_macrophages', 0))
            st.sidebar.metric("С везикулами", stats['coloc_stats'].get('macrophages_with_vesicles', 0))
            st.sidebar.metric("С везикулами %", f"{stats['coloc_stats'].get('macrophage_colocalization_percent', 0.0):.1f}%")
            st.sidebar.metric("Везикул в макрофагах", stats['coloc_stats'].get('vesicles_in_macrophages', 0))
            st.sidebar.metric("Среднее везикул на макрофаг", f"{stats['coloc_stats'].get('mean_vesicles_per_macrophage', 0.0):.2f}")
    
    st.sidebar.divider()
    
    # Кнопки действий - будут обновлены в главном блоке
    st.session_state.save_button = st.sidebar.button("💾 Сохранить", use_container_width=True)
    st.session_state.export_button = st.sidebar.button("📥 Экспорт CSV", use_container_width=True)
    
    if st.button("🔄 Сброс параметров", use_container_width=True):
        st.session_state.params = {
            'min_intensity': 5,  # ⭐ Обновлено
            'collagen_diff_threshold': 0,  # ⭐ Обновлено
            'min_vesicle_size': 2,  # ⭐ Обновлено
            'max_vesicle_size': 200,  # ⭐ Обновлено
            'max_iterations': 15,
            'cluster_threshold_factor': 1.5,
            'igg_percent': 90,
            'collagen_factor': 0.7,
            'vesicles_brightness': 1.0,
            'protein_brightness': 1.7,
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
# ОСНОВНАЯ ПАНЕЛЬ (ПРАВАЯ КОЛОНКА)
# ============================================================================

if st.session_state.current_data is not None:
    st.title("🔬 Анализ везикул")

    # Информация об образце
    metadata = st.session_state.current_data.get('metadata', {})
    protein_name = st.session_state.selected_protein or metadata.get('protein', 'N/A')
    group_name = metadata.get('group', 'N/A')
    sample_name = metadata.get('sample_name', 'N/A').strip()

    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        st.subheader(f"📍 {sample_name}")
    with col2:
        st.caption(f"Белок: {protein_name}")
    with col3:
        st.caption(f"Группа: {group_name}")
    with col4:
        st.caption(f"Размер: {st.session_state.current_data['shape'][0]}×{st.session_state.current_data['shape'][1]}")
    
    # Обработка образца с кешированием по образцу и параметрам
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
        with st.spinner("🔄 Обрабатываю образец..."):
            processed = process_sample(
                st.session_state.current_data['channels'],
                st.session_state.params,
                analyze_macrophages=st.session_state.params['analyze_macrophages'],
                expansion_factor=st.session_state.params['expansion_factor']
            )
            
            # Сохраняем статистику
            st.session_state.stats = processed
            
            # Создаем визуализации
            visuals = create_visualization_images(processed, st.session_state.params)

        st.session_state.processed_cache[sample_cache_key] = {
            'processed': processed,
            'visuals': visuals
        }
        st.session_state.recalculate_segmentation = False

    st.session_state.stats = processed
    
    # Обработка нажатия кнопок сохранения (используем session_state для избежания проблем scope)
    if st.session_state.get('save_button', False):
        with st.spinner("💾 Сохраняю результаты..."):
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
                st.success(f"✅ Результаты сохранены в:\n{output_dir}")
                st.session_state.save_button = False
            except Exception as e:
                st.error(f"❌ Ошибка сохранения: {e}")
    
    if st.session_state.get('export_button', False):
        with st.spinner("📥 Экспортирую статистику..."):
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
                
                st.success(f"✅ Статистика экспортирована в:\n{csv_file}")
                st.session_state.export_button = False
            except Exception as e:
                st.error(f"❌ Ошибка экспорта: {e}")
    
    # Сетка 3x2 для визуализаций
    st.subheader("📊 Результаты анализа")
    
    # Дополнительные каналы
    st.subheader("🔍 Дополнительные каналы")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 🟢 Аутофлуоресценция")
        st.image(visuals['collagen_original'])

    with col2:
        st.markdown("#### 🔵 Белок (оригинал)")
        st.image(visuals['protein_original'])

    with col3:
        st.markdown("#### 🟣 Белок (скорректированный)")
        st.image(visuals['protein_corrected'])

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 1️⃣ Оригинальный композит")
        st.markdown("Ядра (синий) | Аутофлуоресценция (зелёный) | Везикулы (красный)")
        st.image(visuals['composite_original'])
    
    with col2:
        st.markdown("#### 2️⃣ Корректированный композит")
        st.markdown("После вычитания фона")
        st.image(visuals['composite_corrected'])
    
    with col3:
        st.markdown("#### 3️⃣ Сегментация везикул")
        st.markdown(f"Найдено везикул: **{processed['num_vesicles']}**")
        st.image(visuals['colored_overlay'])
        st.markdown("""
            **Цвета везикул:**
            - � **Ярко-циан** – обычная везикула (без колокализации)
            - � **Ярко-маджента** – везикула с колокализацией с белком
            - ⚪ **Белый** – везикула внутри клетки (на коллагене)
            
            *Каждый круг имеет чёрную обводку для лучшей видимости*
            """)
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        st.markdown("#### 4️⃣ оригинальные везикулы")
        st.image(visuals['vesicles_original_red'])
    
    with col5:
        st.markdown("#### 5️⃣ Везикулы с аннотациями")
        st.markdown("Красный — везикулы | Зелёный — контуры")
        st.image(visuals['vesicles_with_contours'])
    
    with col6:
        st.markdown("#### ℹ️ Статистика")
        stats = processed['coloc_stats']
        st.metric("🟢 Везикул всего", stats['total_vesicles'])
        st.metric("В клетке", stats['vesicles_in_cells'])
        st.metric("🔵 С белком", stats['protein_colocalized'])
        st.metric("📊 В клетках, %", f"{stats['percent_in_cells']:.1f}%")
        st.metric("📊 % колокализации", f"{stats['protein_colocalization_percent']:.1f}%")
        if st.session_state.params.get('analyze_macrophages', False):
            st.divider()
            st.write("🧫 **Макрофаги**")
            st.metric("Всего", stats.get('total_macrophages', 0))
            st.metric("С везикулами", stats.get('macrophages_with_vesicles', 0))
            st.metric("% с везикулами", f"{stats.get('macrophage_colocalization_percent', 0.0):.1f}%")
            st.metric("Везикул внутри", stats.get('vesicles_in_macrophages', 0))
            st.metric("Среднее везикул/макрофаг", f"{stats.get('mean_vesicles_per_macrophage', 0.0):.2f}")
    
    # Детальная статистика
    st.divider()
    st.subheader("📈 Детальная статистика")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        stats = processed['coloc_stats']
        st.metric("🟢 Везикул всего", stats['total_vesicles'])
        st.metric("🟡 В клетке", stats['vesicles_in_cells'])
    
    with col2:
        st.metric("🔵 С белком", stats['protein_colocalized'])
        st.metric("📊 % колокализации", f"{stats['protein_colocalization_percent']:.1f}%")
        st.metric("📊 Кол/Клетка, %", f"{stats['colocalization_to_cell_ratio']:.1f}%")
        st.metric("📊 В клетках, %", f"{stats['percent_in_cells']:.1f}%")
    
    with col3:
        st.metric("💾 Средн. интенс. белка", f"{stats['mean_protein_intensity']:.0f}")
        st.metric("💾 Средн. интенс. коллагена", f"{stats['mean_collagen_intensity']:.0f}")


    # Дополнительная строка для макрофагов (если анализ включён)
    if st.session_state.params.get('analyze_macrophages', False):
        st.divider()
        st.subheader("🧫 Детальная статистика макрофагов")
        col4, col5, col6 = st.columns(3)
        with col4:
            stats = processed['coloc_stats']
            st.metric("Всего макрофагов", stats.get('total_macrophages', 0))
            st.metric("Макрофагов с везикулами", stats.get('macrophages_with_vesicles', 0))
        with col5:
            st.metric("% макрофагов с везикулами", f"{stats.get('macrophage_colocalization_percent', 0.0):.1f}%")
            st.metric("Всего везикул в макрофагах", stats.get('vesicles_in_macrophages', 0))
        with col6:
            st.metric("Среднее везикул на макрофаг", f"{stats.get('mean_vesicles_per_macrophage', 0.0):.2f}")
    
    # Информация о каналах из метаданных
    if 'channels' in metadata:
        st.divider()
        st.subheader("ℹ️ Информация о каналах")
        
        channels_info = metadata['channels']
        col1, col2, col3, col4 = st.columns(4)
        
        channel_names = ['Ядра', 'Коллаген', 'Везикулы', 'Белок']
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
    # Главный экран без загруженного образца
    st.title("🔬 Анализ везикул - Streamlit")
    
    st.markdown("""
    ### 👋 Добро пожаловать!
    
    Это интерактивное приложение для анализа везикул в микроскопических изображениях.
    
    **Как начать:**
    1. Выберите группу и образец в левой панели
    2. Нажмите "Загрузить образец"
    3. Настройте параметры анализа
    4. Просмотрите результаты в реальном времени
    
    **Возможности:**
    - 🔬 Сегментация везикул с адаптивными параметрами
    - 🔨 Разделение скоплений методом водораздела
    - 🎯 Анализ колокализации с белком и коллагеном
    - 📊 Интерактивные визуализации
    - 💾 Сохранение результатов
    
    **Начните с левой панели** 👈
    """)
    
    # Проверка наличия данных
    if not get_available_samples(str(st.session_state.preprocessed_dir)):
        st.error("""
        ❌ **Ошибка:** Папка `preprocessed/` не найдена или пуста.
        
        Убедитесь, что:
        - Файлы TIFF находятся в структуре: `preprocessed/group_name/sample_name/channels.tif`
        - Файлы метаданных находятся в структуре: `preprocessed/group_name/sample_name/metadata.json`
        """)
