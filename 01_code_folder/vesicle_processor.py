"""
Модуль для интерактивной обработки везикул в Streamlit приложении.
Использует методы из multi_lif_analyzer_legacy.py.
"""
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import json
import os
from pathlib import Path
from skimage import exposure, measure, morphology, segmentation
from skimage.filters import sobel
from skimage.feature import peak_local_max
from scipy import ndimage
import tifffile
import pandas as pd
from datetime import datetime
import csv
import cv2


class VesicleProcessor:
    """Класс для обработки везикул с адаптацией методов из MultiSampleLifAnalyzer."""
    
    def __init__(self):
        """Инициализация параметров по умолчанию."""
        # Параметры сегментации
        self.min_intensity = 115
        self.collagen_diff_threshold = 15
        self.min_vesicle_size = 5
        self.max_vesicle_size = 50
        
        # Параметры кластеризации
        self.max_iterations = 15
        self.cluster_threshold_factor = 1.5
        
        # Параметры вычитания фона
        self.igg_percent = 90
        self.collagen_factor = 0.7
        self.protein_brightness = 1.7
        
        # Параметры колокализации
        self.protein_threshold = 50
        
    @property
    def _default_params(self):
        """Параметры по умолчанию."""
        return {
            'min_intensity': 115,
            'collagen_diff_threshold': 15,
            'min_vesicle_size': 5,
            'max_vesicle_size': 50,
            'max_iterations': 15,
            'cluster_threshold_factor': 1.5,
            'igg_percent': 90,
            'collagen_factor': 0.7,
            'protein_brightness': 1.7,
            'protein_threshold': 50,
        }
    
    def load_sample(self, preprocessed_dir: Path, protein_name: str, group_name: str, sample_name: str):
        """
        Загружает TIFF файл и метаданные образца.

        Args:
            preprocessed_dir: Path к папке preprocessed/
            protein_name: Название белка (e.g., "FAP", "CD206")
            group_name: Название группы (e.g., "30min_mouse_09")
            sample_name: Название образца (e.g., "9_30_min")

        Returns:
            dict с ключами:
                - channels: dict {0: nuclei, 1: collagen, 2: vesicles, 3: protein}
                - metadata: dict из metadata.json
                - shape: кортеж (height, width)
        """
        sample_path = preprocessed_dir / protein_name / group_name / sample_name
        
        # Загрузим TIFF
        tiff_path = sample_path / "channels.tif"
        if not tiff_path.exists():
            raise FileNotFoundError(f"TIFF не найден: {tiff_path}")
        
        # tifffile возвращает массив (channels, height, width) для многоканального TIFF
        channels_array = tifffile.imread(str(tiff_path))
        
        # Если TIFF имеет формат (height, width, channels), перевернем
        if channels_array.ndim == 3:
            if channels_array.shape[2] == 4:  # (H, W, C)
                channels_array = np.transpose(channels_array, (2, 0, 1))  # (C, H, W)
        
        # Загрузим метаданные
        metadata_path = sample_path / "metadata.json"
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        
        channels_dict = {
            0: channels_array[0],  # nuclei
            1: channels_array[1],  # collagen
            2: channels_array[2],  # vesicles
            3: channels_array[3],  # protein
        }
        
        return {
            'channels': channels_dict,
            'metadata': metadata,
            'shape': channels_array.shape[1:]  # (height, width)
        }
    
    def subtract_collagen_background(self, vesicles_raw, collagen_channel, 
                                     collagen_factor=0.7):
        """
        Вычитание фона коллагена из везикул (pixel-by-pixel).
        
        Args:
            vesicles_raw: numpy array uint16, канал везикул
            collagen_channel: numpy array uint16, канал коллагена
            collagen_factor: коэффициент вычитания (0-1)
        
        Returns:
            corrected: numpy array uint16, после вычитания
        
        Логика:
            Для каждого пикселя (y,x) вычисляется:
            corrected[y,x] = vesicles[y,x] - collagen_factor * collagen[y,x]
        """
        # ⭐ PIXEL-BY-PIXEL вычитание (без нормализации диапазонов)
        vesicles_float = vesicles_raw.astype(np.float32)
        collagen_float = collagen_channel.astype(np.float32)
        
        # Вычитаем коллаген по пикселям
        corrected = vesicles_float - collagen_factor * collagen_float
        
        # Не даем отрицательных значений
        corrected = np.maximum(corrected, 0)
        
        # Приводим обратно к uint16
        corrected = corrected.astype(np.uint16)
        
        return corrected
    
    def subtract_igg_from_protein(self, protein_channel, igg_percent=90):
        """
        Вычитает из белкового канала значение, равное заданному процентилю самого канала.
        """
        protein = protein_channel.astype(np.float32)
        background = np.percentile(protein, igg_percent)
        corrected = protein - background
        corrected = np.maximum(corrected, 0)
        # Нормализуем обратно в uint16
        if corrected.max() > 0:
            corrected = (corrected / corrected.max() * 65535).astype(np.uint16)
        else:
            corrected = corrected.astype(np.uint16)
        return corrected
    
    def segment_vesicles(self, vesicles_channel, collagen_channel, 
                         min_intensity=115, collagen_diff_threshold=15,
                         min_size=5, max_size=50):
        """
        Сегментация везикул с критериями интенсивности и размера.
        
        Логика фильтрации (оба критерия используют pixel-by-pixel сравнение):
        
        1. ⭐ ОСНОВНОЙ КРИТЕРИЙ (интенсивность):
           vesicles[y,x] >= min_intensity
           
           Пример: min_intensity=5
           - Пиксель (50,50) с везикулой интенсивностью 8 → ✅ проходит
           - Пиксель (60,60) с интенсивностью 2 → ❌ не проходит
        
        2. ⭐ ДОПОЛНИТЕЛЬНЫЙ КРИТЕРИЙ (разница с коллагеном):
           vesicles[y,x] - collagen[y,x] >= collagen_diff_threshold
           
           Пример: collagen_diff_threshold=20
           - Пиксель (50,50): везикулы=100, коллаген=70, разница=30 → ✅ (30 >= 20)
           - Пиксель (70,70): везикулы=90, коллаген=75, разница=15 → ❌ (15 < 20)
        
        Args:
            vesicles_channel: numpy array uint16, канал везикул (уже после вычитания коллагена)
            collagen_channel: numpy array uint16, оригинальный канал коллагена
            min_intensity: минимальная интенсивность везикулы (основной критерий)
            collagen_diff_threshold: пороговая разница везикула-коллаген (дополнительный фильтр)
            min_size: минимальный размер объекта в пикселях
            max_size: максимальный размер объекта в пикселях
        
        Returns:
            binary_mask: numpy array bool, маска везикул
        """
        # ⭐ ОСНОВНОЙ КРИТЕРИЙ: пороговая сегментация по интенсивности везикул
        intensity_mask = vesicles_channel >= min_intensity
        
        # ⭐ ДОПОЛНИТЕЛЬНЫЙ КРИТЕРИЙ: везикулы должны быть ярче коллагена
        # (если collagen_diff_threshold=0, этот критерий отключен)
        if collagen_diff_threshold > 0:
            # PIXEL-BY-PIXEL сравнение
            diff = vesicles_channel.astype(int) - collagen_channel.astype(int)
            diff_mask = diff >= collagen_diff_threshold
            combined = intensity_mask & diff_mask
        else:
            # Если пороговая разница отключена, используем только интенсивность
            combined = intensity_mask
        
        # Удаляем очень маленькие объекты
        binary = morphology.remove_small_objects(combined, min_size=min_size)
        
        # Удаляем очень большие объекты
        labeled = measure.label(binary)
        props = measure.regionprops(labeled)
        
        for prop in props:
            if prop.area > max_size:
                binary[labeled == prop.label] = False
        
        return binary
    
    def segment_and_cluster_vesicles(self, vesicles_channel, collagen_channel,
                                     min_intensity=115, collagen_diff_threshold=15,
                                     min_size=5, max_size=50,
                                     max_iterations=15, cluster_threshold_factor=1.5):
        """
        Улучшенная сегментация везикул с быстрым режимом для больших плотных кадров.

        Для слабых и средних полей работает полный watershed-пайплайн.
        Для очень плотных кадров ограничивает число повторных локальных разбиений,
        чтобы не перегружать слабые компьютеры.
        """
        intensity_mask = vesicles_channel >= min_intensity

        if collagen_diff_threshold > 0:
            diff = vesicles_channel.astype(np.int32) - collagen_channel.astype(np.int32)
            base_mask = intensity_mask & (diff >= collagen_diff_threshold)
        else:
            base_mask = intensity_mask

        base_mask = morphology.remove_small_objects(base_mask, min_size=min_size)
        base_mask = morphology.binary_closing(base_mask, morphology.disk(1))
        base_mask = morphology.remove_small_holes(base_mask, area_threshold=max(4, min_size))

        if not np.any(base_mask):
            empty = np.zeros_like(base_mask, dtype=np.int32)
            return base_mask.astype(bool), empty, 0

        distance_transform = ndimage.distance_transform_edt(base_mask)

        intensity_float = vesicles_channel.astype(np.float32)
        inside_values = intensity_float[base_mask]
        lower = float(np.percentile(inside_values, 5)) if inside_values.size else float(intensity_float.min())
        upper = float(np.percentile(inside_values, 95)) if inside_values.size else float(intensity_float.max())
        if upper <= lower:
            upper = float(intensity_float.max()) if intensity_float.max() > lower else lower + 1.0

        intensity_norm = np.clip((intensity_float - lower) / (upper - lower), 0.0, 1.0)
        intensity_norm = np.where(base_mask, intensity_norm, 0.0)

        if distance_transform.max() > 0:
            distance_norm = distance_transform / distance_transform.max()
        else:
            distance_norm = np.zeros_like(distance_transform, dtype=np.float32)

        marker_score = 0.65 * distance_norm + 0.35 * intensity_norm
        marker_score = ndimage.gaussian_filter(marker_score.astype(np.float32), sigma=1.0)
        marker_score = np.where(base_mask, marker_score, 0.0)

        markers = self.find_peak_markers(
            marker_score,
            base_mask,
            min_distance=3,
            threshold_rel=np.clip(0.2 * cluster_threshold_factor, 0.05, 0.35),
            threshold_abs=0.1 * float(marker_score.max()) if marker_score.max() > 0 else None,
            min_peaks=1
        )

        if markers.max() == 0:
            markers = self.find_peak_markers(
                marker_score,
                base_mask,
                min_distance=2,
                threshold_rel=np.clip(0.1 * cluster_threshold_factor, 0.03, 0.2),
                threshold_abs=0.05 * float(marker_score.max()) if marker_score.max() > 0 else None,
                min_peaks=1
            )

        if markers.max() == 0:
            markers = measure.label(base_mask)

        labeled = segmentation.watershed(-distance_transform, markers, mask=base_mask)

        foreground_pixels = int(base_mask.sum())
        initial_components = int(measure.label(base_mask).max())
        dense_sample = foreground_pixels > 25000 or initial_components > 1500
        very_dense_sample = foreground_pixels > 60000

        # Определяем, сколько проходов кластеризации делать
        if very_dense_sample:
            max_passes = min(max_iterations, 5)
        elif dense_sample:
            max_passes = min(max_iterations,10)   # для плотных – не более 3
        else:
            max_passes = max_iterations            # для обычных – сколько задал пользователь

        # Ограничим общее число проходов, чтобы не зависнуть
        max_passes = min(max_passes, 50)

        # Итеративное разделение (один цикл, без дублирования)
        for _ in range(max_passes):
            # Используем тот же iterative_cluster_splitting, но с улучшенными параметрами
            # Передаём current_labeled, а не исходный
            labeled = self.iterative_cluster_splitting(
                labeled=labeled,
                vesicles_channel=vesicles_channel,
                collagen_channel=collagen_channel,
                base_mask=base_mask,
                distance_transform=distance_transform,
                min_size=min_size,
                max_size=max_size,
                max_iterations=1,                     # за один внешний проход делаем один внутренний шаг
                cluster_threshold_factor=cluster_threshold_factor,
                fast_mode=dense_sample,
                max_clusters_to_split=100 if not dense_sample else 30   # больше кластеров за раз
            )

        # После всех проходов – удаляем только экстремально большие объекты (например > max_size*3)
        # Это нужно, чтобы отбросить артефакты, которые невозможно разделить.
        props = measure.regionprops(labeled)
        for prop in props:
            if prop.area > max_size * 3:
                labeled[labeled == prop.label] = 0

        # Объединяем слишком маленькие объекты
        labeled = self._merge_tiny_objects(labeled, min_size)
        labeled = segmentation.relabel_sequential(labeled)[0].astype(np.int32)

        num_vesicles = int(labeled.max())
        return base_mask.astype(bool), labeled, num_vesicles
    
    @staticmethod
    def find_peak_markers(score, binary_mask, min_distance=3, threshold_rel=0.2,
                          threshold_abs=None, min_peaks=1):
        """
        Находит маркеры по пикам в комбинированной карте score.

        Использует peak_local_max с более чувствительными параметрами и
        ограничивает поиск только внутри бинарной маски.
        """
        if score.size == 0 or not np.any(binary_mask):
            return np.zeros_like(binary_mask, dtype=np.int32)

        score = np.asarray(score, dtype=np.float32)
        score = np.where(binary_mask, score, 0.0)

        if threshold_abs is None:
            threshold_abs = 0.1 * float(score.max()) if score.max() > 0 else 0.0

        peaks = peak_local_max(
            score,
            min_distance=min_distance,
            threshold_rel=threshold_rel,
            threshold_abs=threshold_abs,
            exclude_border=False,
            labels=binary_mask.astype(np.uint8)
        )

        if peaks.size == 0 and min_distance > 1:
            peaks = peak_local_max(
                score,
                min_distance=max(1, min_distance - 1),
                threshold_rel=max(0.01, threshold_rel * 0.5),
                threshold_abs=0.05 * float(score.max()) if score.max() > 0 else 0.0,
                exclude_border=False,
                labels=binary_mask.astype(np.uint8)
            )

        markers = np.zeros_like(binary_mask, dtype=np.int32)
        for idx, (row, col) in enumerate(peaks, start=1):
            markers[row, col] = idx

        if markers.max() < min_peaks and np.any(binary_mask):
            local_max = ndimage.maximum_filter(score, size=max(3, min_distance * 2)) == score
            local_max &= binary_mask
            markers = measure.label(local_max.astype(bool)).astype(np.int32)

        return markers

    @staticmethod
    def _find_local_maxima_advanced(distance_transform, vesicles_channel, binary,
                                    threshold_factor=1.5, min_distance=2):
        """
        Совместимость со старым кодом.

        Новый код использует find_peak_markers() по комбинированной карте score.
        """
        score = np.zeros_like(distance_transform, dtype=np.float32)
        if distance_transform.max() > 0:
            score += 0.7 * (distance_transform / distance_transform.max())

        vesicles_channel = vesicles_channel.astype(np.float32)
        if np.any(binary):
            inside = vesicles_channel[binary]
            lo = float(np.percentile(inside, 5)) if inside.size else float(vesicles_channel.min())
            hi = float(np.percentile(inside, 95)) if inside.size else float(vesicles_channel.max())
            if hi <= lo:
                hi = lo + 1.0
            intensity_norm = np.clip((vesicles_channel - lo) / (hi - lo), 0.0, 1.0)
            score += 0.3 * np.where(binary, intensity_norm, 0.0)

        return self.find_peak_markers(
            score,
            binary,
            min_distance=min_distance,
            threshold_rel=np.clip(0.2 * threshold_factor, 0.05, 0.35),
            threshold_abs=0.1 * float(score.max()) if score.max() > 0 else None,
            min_peaks=1
        )
    
    @staticmethod
    def _find_markers_fallback(distance_transform, binary, threshold_factor=1.5):
        """
        Резервный метод поиска маркеров, если peak_local_max не дал результатов.
        """
        if distance_transform.size == 0 or not np.any(binary):
            return np.zeros_like(binary, dtype=np.int32)

        max_dist = float(distance_transform.max())
        positive = distance_transform[distance_transform > 0]
        mean_dist = float(positive.mean()) if positive.size else 0.0
        threshold = min(mean_dist * threshold_factor, 0.4 * max_dist) if max_dist > 0 else 0.0

        local_max = ndimage.maximum_filter(distance_transform, size=5) == distance_transform
        local_max = (distance_transform > threshold) & local_max & binary

        markers = measure.label(local_max).astype(np.int32)
        return markers

    @staticmethod
    def iterative_cluster_splitting(labeled, vesicles_channel, collagen_channel, base_mask,
                                     distance_transform, min_size, max_size, max_iterations,
                                     cluster_threshold_factor=1.5, fast_mode=False,
                                     max_clusters_to_split=20):
        """
        Итеративно разделяет только самые большие кластеры.

        Оптимизации для слабых компьютеров:
        - ограничение числа кластеров за проход
        - отсутствие relabel_sequential на каждом цикле
        - укороченный поиск в fast_mode
        """
        if labeled.size == 0 or not np.any(labeled) or max_iterations <= 0:
            return labeled

        refined = labeled.astype(np.int32).copy()
        next_label = int(refined.max()) + 1
        large_area_threshold = max(1, int(max_size))

        for _ in range(max_iterations):
            region_props = [p for p in measure.regionprops(refined) if p.area > large_area_threshold]
            if not region_props:
                break

            region_props = sorted(region_props, key=lambda p: p.area, reverse=True)
            changed = False

            for prop in region_props:
                #if fast_mode and prop.area > large_area_threshold * 8:
                    #continue

                current_mask = refined == prop.label
                if current_mask.sum() <= large_area_threshold:
                    continue

                min_row, min_col, max_row, max_col = prop.bbox
                pad = 2 if fast_mode else max(3, int(np.ceil(np.sqrt(prop.area) / 12.0)))
                min_row = max(0, min_row - pad)
                min_col = max(0, min_col - pad)
                max_row = min(refined.shape[0], max_row + pad)
                max_col = min(refined.shape[1], max_col + pad)

                patch_mask = current_mask[min_row:max_row, min_col:max_col]
                if not np.any(patch_mask):
                    continue

                patch_dt = distance_transform[min_row:max_row, min_col:max_col]
                patch_vesicles = vesicles_channel[min_row:max_row, min_col:max_col].astype(np.float32)

                patch_inside = patch_vesicles[patch_mask]
                lo = float(np.percentile(patch_inside, 5)) if patch_inside.size else float(patch_vesicles.min())
                hi = float(np.percentile(patch_inside, 95)) if patch_inside.size else float(patch_vesicles.max())
                if hi <= lo:
                    hi = lo + 1.0

                patch_intensity = np.clip((patch_vesicles - lo) / (hi - lo), 0.0, 1.0)
                patch_intensity = np.where(patch_mask, patch_intensity, 0.0)

                if patch_dt.max() > 0:
                    patch_dt_norm = patch_dt / patch_dt.max()
                else:
                    patch_dt_norm = np.zeros_like(patch_dt, dtype=np.float32)

                patch_score = 0.3 * patch_dt_norm + 0.7 * patch_intensity
                if not fast_mode or patch_score.size < 4096:
                    patch_score = ndimage.gaussian_filter(patch_score.astype(np.float32), sigma=1.0)
                patch_score = np.where(patch_mask, patch_score, 0.0)

                markers = VesicleProcessor.find_peak_markers(
                    patch_score,
                    patch_mask,
                    min_distance=1,
                    threshold_rel=np.clip(0.1 * cluster_threshold_factor, 0.03, 0.2),
                    threshold_abs=0.05 * float(patch_score.max()) if patch_score.max() > 0 else None,
                    min_peaks=2
                )

                if int(markers.max()) < 2:
                    continue

                sub_labels = segmentation.watershed(-patch_dt, markers, mask=patch_mask).astype(np.int32)
                if int(sub_labels.max()) < 2:
                    continue

                refined_patch = refined[min_row:max_row, min_col:max_col]
                refined_patch[patch_mask] = 0

                # Список для хранения меток новых фрагментов, которые ещё слишком большие
                new_large_patches = []

                for sub_label in range(1, int(sub_labels.max()) + 1):
                    sub_mask = sub_labels == sub_label
                    sub_area = np.sum(sub_mask)
                    if sub_area > max_size:
                        # Запоминаем этот фрагмент для повторного разделения в следующей итерации
                        # Сохраняем его bounding box и маску в глобальных координатах
                        coords = np.where(sub_mask)
                        if len(coords[0]) > 0:
                            min_r = min_row + np.min(coords[0])
                            max_r = min_row + np.max(coords[0]) + 1
                            min_c = min_col + np.min(coords[1])
                            max_c = min_col + np.max(coords[1]) + 1
                            new_large_patches.append({
                                'bbox': (min_r, min_c, max_r, max_c),
                                'area': sub_area
                            })
                    # Присваиваем новую метку (даже если большой – потом он будет переразделён)
                    refined_patch[sub_mask] = next_label
                    next_label += 1

                changed = True

            if not changed:
                break

        return segmentation.relabel_sequential(refined)[0].astype(np.int32)

    @staticmethod
    def _iterative_cluster_split(labeled, distance_transform, binary, max_size, max_iterations):
        """
        Совместимость со старым кодом.
        """
        return VesicleProcessor.iterative_cluster_splitting(
            labeled=labeled,
            vesicles_channel=np.where(binary, distance_transform, 0).astype(np.uint16),
            collagen_channel=np.zeros_like(distance_transform, dtype=np.uint16),
            base_mask=binary,
            distance_transform=distance_transform,
            min_size=1,
            max_size=max_size,
            max_iterations=max_iterations,
            cluster_threshold_factor=1.5
        )
    
    @staticmethod
    def _merge_tiny_objects(labeled, min_size):
        """
        Объединяет слишком маленькие объекты с ближайшими соседями.

        Если соседей нет, объект удаляется.
        """
        if labeled.size == 0 or not np.any(labeled):
            return labeled

        merged = labeled.astype(np.int32).copy()
        props = [p for p in measure.regionprops(merged) if p.area < max(1, min_size)]

        if not props:
            return segmentation.relabel_sequential(merged)[0].astype(np.int32)

        for prop in sorted(props, key=lambda p: p.area):
            current_label = prop.label
            current_mask = merged == current_label
            if not np.any(current_mask):
                continue

            dilated = ndimage.binary_dilation(current_mask, iterations=2)
            neighbor_labels = np.unique(merged[dilated & ~current_mask])
            neighbor_labels = neighbor_labels[(neighbor_labels > 0) & (neighbor_labels != current_label)]

            if neighbor_labels.size > 0:
                border_counts = []
                for neighbor_label in neighbor_labels:
                    neighbor_mask = merged == neighbor_label
                    contact = np.sum(ndimage.binary_dilation(current_mask, iterations=1) & neighbor_mask)
                    border_counts.append((contact, int(neighbor_label)))
                border_counts.sort(reverse=True)
                target_label = border_counts[0][1]
                merged[current_mask] = target_label
            else:
                merged[current_mask] = 0

        return segmentation.relabel_sequential(merged)[0].astype(np.int32)
    
    def analyze_colocalization(self, vesicles_labeled, collagen_channel, protein_channel,
                           nuclei_channel=None, protein_threshold=50, collagen_percentile=50, nuclei_percentile=50):
        total_vesicles = len(np.unique(vesicles_labeled)) - 1
        if total_vesicles == 0:
            return {
                'total_vesicles': 0,
                'collagen_colocalized': 0,
                'protein_colocalized': 0,
                'protein_colocalization_percent': 0.0,
                'vesicles_in_cells': 0,
                'colocalization_to_cell_ratio': 0.0,
                'percent_in_cells': 0.0,
                'mean_protein_intensity': 0.0,
                'mean_collagen_intensity': 0.0,
                'mean_vesicle_intensity': 0.0,
            }

        collagen_threshold = np.percentile(collagen_channel, collagen_percentile)
        nuclei_threshold = np.percentile(nuclei_channel, nuclei_percentile) if nuclei_channel is not None else None
        props = measure.regionprops(vesicles_labeled, intensity_image=protein_channel)

        collagen_colocalized = 0
        protein_colocalized = 0
        vesicles_in_cells = 0
        mean_protein_intensities = []
        mean_collagen_intensities = []

        for prop in props:
            mask = vesicles_labeled == prop.label

            # Коллаген
            collagen_intensity = collagen_channel[mask].mean()
            mean_collagen_intensities.append(collagen_intensity)
            if collagen_intensity > collagen_threshold:
                collagen_colocalized += 1

            # Белок
            protein_intensity = protein_channel[mask].mean()
            mean_protein_intensities.append(protein_intensity)

            # Определяем, находится ли везикула в клетке
            in_cell = (collagen_intensity > collagen_threshold)
            if nuclei_channel is not None and nuclei_threshold is not None:
            # Здесь используем СРЕДНЮЮ интенсивность ядер, а не any > 0
                nuclei_intensity = nuclei_channel[mask].mean()
                in_cell = in_cell or (nuclei_intensity > nuclei_threshold)   # любая ненулевая интенсивность ядра

            if in_cell:
                vesicles_in_cells += 1
                if protein_intensity > protein_threshold:
                    protein_colocalized += 1

        protein_percent = (protein_colocalized / total_vesicles * 100) if total_vesicles > 0 else 0
        percent_in_cells = (vesicles_in_cells / total_vesicles * 100) if total_vesicles > 0 else 0.0

        if vesicles_in_cells > 0:
            colocalization_to_cell_ratio = (protein_colocalized / vesicles_in_cells) * 100
        else:
            colocalization_to_cell_ratio = 0.0

        return {
            'total_vesicles': total_vesicles,
            'collagen_colocalized': collagen_colocalized,
            'protein_colocalized': protein_colocalized,
            'protein_colocalization_percent': protein_percent,
            'vesicles_in_cells': vesicles_in_cells,
            'colocalization_to_cell_ratio': colocalization_to_cell_ratio,
            'percent_in_cells': percent_in_cells,
            'mean_protein_intensity': float(np.mean(mean_protein_intensities)) if mean_protein_intensities else 0.0,
            'mean_collagen_intensity': float(np.mean(mean_collagen_intensities)) if mean_collagen_intensities else 0.0,
            'mean_vesicle_intensity': float(np.mean(mean_protein_intensities)) if mean_protein_intensities else 0.0,
        }
    
    @staticmethod
    def create_composite_rgb(nuclei, collagen, vesicles, protein=None):
        # Преобразуем 16-битные каналы в 8-битные путём сдвига битов (деление на 256)
        def to_uint8(channel):
            # Если канал уже uint8, просто вернуть
            if channel.dtype == np.uint8:
                return channel
            # Иначе сдвиг вправо на 8 бит (деление на 256) и приведение к uint8
            return (channel >> 8).astype(np.uint8)
        
        nuclei_8 = to_uint8(nuclei)
        vesicles_8 = to_uint8(vesicles)
        
        if protein is not None:
            protein_8 = to_uint8(protein)
        else:
            protein_8 = to_uint8(collagen)
        
        rgb = np.zeros((nuclei_8.shape[0], nuclei_8.shape[1], 3), dtype=np.uint8)
        rgb[:, :, 0] = vesicles
        rgb[:, :, 1] = protein_8
        rgb[:, :, 2] = nuclei
        return rgb
    
    @staticmethod
    def create_segmentation_overlay(original_rgb, segmentation_labeled, color_colocalized=(255, 255, 0)):
        """
        Создание overlay маски сегментации на оригинальное изображение.
        
        Args:
            original_rgb: numpy array (H, W, 3) uint8
            segmentation_labeled: numpy array int с метками везикул
            color_colocalized: tuple RGB для везикул с колокализацией
        
        Returns:
            numpy array (H, W, 3) uint8 с overlay
        """
        overlay = original_rgb.copy().astype(np.float32)
        
        # Контуры везикул
        for label in np.unique(segmentation_labeled):
            if label == 0:  # Пропускаем фон
                continue
            
            mask = segmentation_labeled == label
            contours = sobel(mask.astype(float))
            contour_mask = contours > 0
            
            # ⭐ Правильное присваивание для каждого канала отдельно
            overlay[contour_mask, 0] = overlay[contour_mask, 0] * 0.7 + color_colocalized[0] * 0.3
            overlay[contour_mask, 1] = overlay[contour_mask, 1] * 0.7 + color_colocalized[1] * 0.3
            overlay[contour_mask, 2] = overlay[contour_mask, 2] * 0.7 + color_colocalized[2] * 0.3
        
        return np.clip(overlay, 0, 255).astype(np.uint8)
    
    
    def create_vesicles_with_contours(self, vesicles_corrected, segmentation_labeled):
        """
        Создает RGB изображение везикул красным цветом с зелёными контурами аннотаций.
        
        Args:
            vesicles_corrected: numpy array uint16, корректированный канал везикул
            segmentation_labeled: numpy array uint16, меченное изображение везикул
        
        Returns:
            rgb_image: numpy array uint8 shape (H, W, 3), RGB изображение
        """
        # Нормализуем в диапазон 0-255
        vesicles_norm = (vesicles_corrected.astype(np.float32) / vesicles_corrected.max() * 255).astype(np.uint8)
        
        # Создаем RGB изображение (черный фон)
        rgb_image = np.zeros((vesicles_norm.shape[0], vesicles_norm.shape[1], 3), dtype=np.uint8)
        
        # Красный канал = интенсивность везикул
        rgb_image[:, :, 0] = vesicles_norm  # Red channel
        # Green и Blue остаются черными (0)
        
        # Рисуем зеленые контуры везикул
        props = measure.regionprops(segmentation_labeled)
        for prop in props:
            # Получаем центроид и радиус
            y, x = prop.centroid
            radius = int(np.sqrt(prop.area / np.pi)) + 2
            
            # Рисуем зелёный круг (контур)
            cv2.circle(rgb_image, (int(x), int(y)), radius, (0, 255, 0), 2)
        
        return rgb_image
    
    @staticmethod
    def extract_vesicle_gallery(vesicles_channel, segmentation_labeled, 
                               gallery_size=32, max_vesicles=25):
        """
        Извлечение галереи отдельных везикул.
        
        Args:
            vesicles_channel: numpy array uint16
            segmentation_labeled: numpy array int
            gallery_size: размер квадрата для каждой везикулы
            max_vesicles: максимум везикул в галерее
        
        Returns:
            numpy array (gallery_height, gallery_width, 3) uint8 - монтаж везикул
        """
        props = measure.regionprops(segmentation_labeled)
        
        # Ограничиваем количество
        props = sorted(props, key=lambda p: p.area, reverse=True)[:max_vesicles]
        
        gallery_patches = []
        for prop in props:
            min_row, min_col, max_row, max_col = prop.bbox
            patch = vesicles_channel[min_row:max_row, min_col:max_col]
            
            # Обрезаем/паддим до gallery_size x gallery_size
            h, w = patch.shape
            if h > gallery_size or w > gallery_size:
                # Обрезаем от центра
                start_h = max(0, (h - gallery_size) // 2)
                start_w = max(0, (w - gallery_size) // 2)
                patch = patch[start_h:start_h+gallery_size, start_w:start_w+gallery_size]
            
            if patch.shape != (gallery_size, gallery_size):
                padded = np.zeros((gallery_size, gallery_size), dtype=patch.dtype)
                ph, pw = patch.shape
                padded[:ph, :pw] = patch
                patch = padded
            
            # Нормализуем в uint8
            patch_norm = (patch.astype(float) / patch.max() * 255).astype(np.uint8) if patch.max() > 0 else patch.astype(np.uint8)
            gallery_patches.append(patch_norm)
        
        if not gallery_patches:
            return np.zeros((gallery_size, gallery_size*5, 3), dtype=np.uint8)
        
        # Создаем монтаж (5 везикул в ряду)
        cols = 5
        rows = (len(gallery_patches) + cols - 1) // cols
        
        gallery = np.zeros((rows * gallery_size, cols * gallery_size, 3), dtype=np.uint8)
        
        for idx, patch in enumerate(gallery_patches):
            row_idx = idx // cols
            col_idx = idx % cols
            
            # Конвертим в RGB если нужно
            if patch.ndim == 2:
                patch_rgb = np.stack([patch, patch, patch], axis=-1)
            else:
                patch_rgb = patch
            
            gallery[
                row_idx*gallery_size:(row_idx+1)*gallery_size,
                col_idx*gallery_size:(col_idx+1)*gallery_size
            ] = patch_rgb
        
        return gallery
    
    def export_statistics_csv(self, stats, output_path, sample_metadata=None):
        """
        Экспортирует статистику в CSV файл.
        
        Args:
            stats: dict со статистикой (результат analyze_colocalization)
            output_path: Path или str для сохранения CSV
            sample_metadata: dict опциональных метаданных образца
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Подготавливаем данные
        export_data = {
            'Timestamp': datetime.now().isoformat(),
            'Sample': sample_metadata.get('sample_name', 'Unknown') if sample_metadata else 'Unknown',
            'Group': sample_metadata.get('group', 'Unknown') if sample_metadata else 'Unknown',
            'Total Vesicles': stats['total_vesicles'],
            'Vesicles_in_cells': stats['vesicles_in_cells'],
            'Protein Colocalized': stats['protein_colocalized'],
            'Colocalization %': f"{stats['protein_colocalization_percent']:.2f}",
            'Colocalization to Cell Ratio %': f"{stats['colocalization_to_cell_ratio']:.2f}",
            'Percent in Cells': f"{stats['percent_in_cells']:.2f}",
            'Mean Protein Intensity': f"{stats['mean_protein_intensity']:.2f}",
            'Mean Collagen Intensity': f"{stats['mean_collagen_intensity']:.2f}",
            'Mean Vesicle Intensity': f"{stats['mean_vesicle_intensity']:.2f}",
            'Total Macrophages': stats.get('total_macrophages', 0),
            'Macrophages with Vesicles': stats.get('macrophages_with_vesicles', 0),
            'Macrophage Colocalization %': f"{stats.get('macrophage_colocalization_percent', 0.0):.2f}",
            'Vesicles in Macrophages': stats.get('vesicles_in_macrophages', 0),
            'Mean Vesicles per Macrophage': f"{stats.get('mean_vesicles_per_macrophage', 0.0):.2f}",
        }
        
        # Записываем в CSV
        df = pd.DataFrame([export_data])
        df.to_csv(output_path, index=False)
        
        return output_path
    
    def save_analysis_state(self, output_dir, sample_metadata, params, stats, 
                           processed_data, visualizations):
        """
        Сохраняет полное состояние анализа (параметры, статистика, изображения).
        
        Args:
            output_dir: Path или str для сохранения результатов
            sample_metadata: dict метаданных образца
            params: dict параметров анализа
            stats: dict статистики
            processed_data: dict обработанных каналов
            visualizations: dict визуализаций
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Создаем подпапку для образца
        sample_slug = sample_metadata.get('sample_slug', 'sample')
        sample_dir = output_dir / sample_slug
        sample_dir.mkdir(exist_ok=True)
        
        # 1. Сохраняем параметры
        params_file = sample_dir / 'analysis_params.json'
        with open(params_file, 'w') as f:
            json.dump(params, f, indent=2)
        
        # 2. Сохраняем статистику
        stats_file = sample_dir / 'statistics.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        # 3. Сохраняем метаданные
        metadata_file = sample_dir / 'sample_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(sample_metadata, f, indent=2)
        
        # 4. Сохраняем визуализации как PNG
        try:
            from PIL import Image
            
            for vis_name, vis_image in visualizations.items():
                if vis_image is not None:
                    img = Image.fromarray(vis_image)
                    img.save(sample_dir / f'{vis_name}.png')
        except ImportError:
            pass  # PIL не установлен
        
        # 5. Сохраняем корректированные каналы как TIFF
        channels_tiff = sample_dir / 'corrected_channels.tif'
        corrected_channels = np.stack([
            processed_data['vesicles_corrected'],
            processed_data['protein_corrected'],
        ])
        tifffile.imwrite(str(channels_tiff), corrected_channels)
        
        # 6. Сохраняем маску сегментации
        segmentation_tiff = sample_dir / 'segmentation_mask.tif'
        tifffile.imwrite(str(segmentation_tiff), processed_data['vesicles_labeled'].astype(np.uint16))
        
        return sample_dir
    
    @staticmethod
    def merge_statistics_csv(stats_files, output_path):
        """
        Объединяет несколько CSV файлов со статистикой в один.
        
        Args:
            stats_files: список Path объектов или строк путей к CSV файлам
            output_path: Path или str для сохранения объединенного файла
        """
        all_data = []
        
        for csv_file in stats_files:
            df = pd.read_csv(csv_file)
            all_data.append(df)
        
        merged_df = pd.concat(all_data, ignore_index=True)
        merged_df.to_csv(output_path, index=False)
        
        return output_path
    
    def create_vesicles_original_red(self, vesicles_raw):
        """
        Отображает оригинальный канал везикул в красном цвете.
        
        Args:
            vesicles_raw: numpy array uint16, оригинальный канал везикул
        
        Returns:
            rgb_image: numpy array uint8 shape (H, W, 3), красное RGB изображение
        """
        # Нормализуем в диапазон 0-255
        vesicles_norm = (vesicles_raw.astype(np.float32) / vesicles_raw.max() * 255).astype(np.uint8)
        
        # Создаем RGB изображение (черный фон)
        rgb_image = np.zeros((vesicles_norm.shape[0], vesicles_norm.shape[1], 3), dtype=np.uint8)
        
        # Красный канал = интенсивность везикул
        rgb_image[:, :, 0] = vesicles_norm  # Red channel
        # Green и Blue остаются черными (0)
        return rgb_image
    def create_collagen_original_green(self, collagen_raw):
        """Коллаген – зелёный цвет."""
        collagen_norm = (collagen_raw.astype(np.float32) / collagen_raw.max() * 255).astype(np.uint8)
        rgb_image = np.zeros((collagen_norm.shape[0], collagen_norm.shape[1], 3), dtype=np.uint8)
        rgb_image[:, :, 1] = collagen_norm   # зелёный канал
        return rgb_image


    @staticmethod
    def normalize_to_rgb(img, cmap='gray'):
        """Преобразует одноканальное изображение в RGB (серое)."""
        img = img.astype(np.float32)
        if img.max() > 0:
            img = (img - img.min()) / (img.max() - img.min()) * 255
        img = np.clip(img, 0, 255).astype(np.uint8)
        rgb = np.stack([img, img, img], axis=-1)
        return rgb
    
    @staticmethod
    def adjust_brightness(channel, brightness_factor=1.0):
        adjusted = channel.astype(np.float32) * brightness_factor
        adjusted = np.clip(adjusted, 0, 65535).astype(np.uint16)
        return adjusted
    

    

    def classify_vesicles(self, vesicles_labeled, protein_channel, collagen_channel,
                      nuclei_channel=None, protein_threshold=50, collagen_percentile=50, nuclei_percentile=50, macrophage_labels=None):
        if vesicles_labeled is None:
            return np.array([]), {}

        collagen_threshold = np.percentile(collagen_channel, collagen_percentile)
        nuclei_threshold = np.percentile(nuclei_channel, nuclei_percentile) if nuclei_channel is not None else None
        labels = np.unique(vesicles_labeled)
        labels = labels[labels != 0]

        vesicle_classes = {}
        classification_mask = np.zeros_like(vesicles_labeled, dtype=np.uint8)

        for label in labels:
            mask = vesicles_labeled == label
            mean_protein = protein_channel[mask].mean()
            mean_collagen = collagen_channel[mask].mean()

            in_cell = (mean_collagen > collagen_threshold)
            if nuclei_channel is not None and nuclei_threshold is not None:
                mean_nuclei = nuclei_channel[mask].mean()
                in_cell = in_cell or (mean_nuclei > nuclei_threshold)

            # Белковая колокализация возможна только внутри клетки
            with_protein = (mean_protein > protein_threshold) and in_cell

            in_macrophage = False
            if macrophage_labels is not None and np.max(macrophage_labels) > 0:
                if np.any(macrophage_labels[mask] > 0):
                    in_macrophage = True

            if in_macrophage:
                cls = 3         #в макрофаге
            elif with_protein:
                cls = 1         #Колокализирована
            elif in_cell:
                cls = 2         #В клетке
            else:
                cls = 0         # вне клеток

            vesicle_classes[label] = cls
            classification_mask[mask] = cls

        return classification_mask, vesicle_classes
    

    def create_colored_vesicle_overlay(self, composite_rgb, vesicles_labeled, vesicle_classes,
                                    colors={0: (0, 255, 255),    # ярко-циан
                                            1: (255, 0, 255),    # ярко-маджента
                                            2: (255, 255, 255)}, # белый
                                    thickness=3,
                                    outer_color=(0, 0, 0),
                                    outer_thickness=1):
        """
        Рисует яркие, контрастные цветные круги вокруг везикул на композитном изображении.
        
        Особенности:
        - Используются максимально контрастные цвета (циан, маджента, белый)
        - Каждый круг имеет чёрную обводку (1 пиксель) для видимости на светлых участках
        - Большие радиусы чтобы избежать наложения в плотных скоплениях
        - Прямое присваивание пикселей (без смешивания с прозрачностью)
        
        Args:
            composite_rgb: исходное RGB изображение
            vesicles_labeled: маска с метками везикул
            vesicle_classes: словарь {label: class}
            colors: словарь класса -> цвет RGB
                    0: ярко-циан (0, 255, 255) - обычная везикула
                    1: ярко-маджента (255, 0, 255) - везикула с белком
                    2: белый (255, 255, 255) - везикула в клетке
            thickness: толщина основного контура (рекомендуется 3)
            outer_color: цвет внешней обводки RGB (чёрный по умолчанию)
            outer_thickness: толщина внешней обводки (рекомендуется 1)
        
        Returns:
            overlay: изображение с нарисованными контрастными кругами
        """
        overlay = composite_rgb.astype(np.uint8).copy()
        
        for label, cls in vesicle_classes.items():
            mask = vesicles_labeled == label
            # Находим центр и радиус
            props = measure.regionprops(mask.astype(int))
            if not props:
                continue
            prop = props[0]
            y, x = prop.centroid
            # Увеличиваем радиус на 3-4 пикселя больше, чтобы избежать слияния
            radius = int(np.sqrt(prop.area / np.pi)) + 4
            
            center = (int(x), int(y))
            
            # Выбираем цвет (RGB в исходном формате)
            color_rgb = colors.get(cls, (255, 255, 255))
            
            # Сначала рисуем чёрную обводку (внешняя обводка)
            outer_color_bgr = (outer_color[2], outer_color[1], outer_color[0])
            cv2.circle(overlay, center, radius + outer_thickness, outer_color_bgr, 
                      outer_thickness, lineType=cv2.LINE_AA)
            
            # Затем рисуем основной цветной круг
            # OpenCV использует BGR, поэтому переворачиваем RGB -> BGR
            color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
            cv2.circle(overlay, center, radius, color_bgr, thickness, lineType=cv2.LINE_AA)
        
        return overlay

    def add_scale_bar(self, image, scale_length_pixels=100, scale_text="100 μm", 
                  bar_height=6, position='bottom_right', margin=25):
        """Добавляет белую шкалу с текстом внизу (без черного фона)."""
        try:
            if image is None:
                return None
                
            # Конвертируем в PIL Image если это numpy array
            if isinstance(image, np.ndarray):
                if len(image.shape) == 3 and image.shape[2] == 3:
                    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                else:
                    pil_image = Image.fromarray(image).convert('RGB')
            else:
                pil_image = image.copy()
            
            draw = ImageDraw.Draw(pil_image)
            width, height = pil_image.size
            
            if position == 'bottom_right':
                x_start = width - scale_length_pixels - margin
                x_end = width - margin
                y_bar = height - margin - bar_height
                y_text = y_bar + bar_height + 5
                text_x = x_start + (scale_length_pixels - 40) // 2
            else:  # bottom_center
                x_start = (width - scale_length_pixels) // 2
                x_end = x_start + scale_length_pixels
                y_bar = height - margin - bar_height
                y_text = y_bar + bar_height + 5
                text_x = x_start + (scale_length_pixels - 40) // 2
            
            # Рисуем белую шкалу
            draw.rectangle([x_start, y_bar, x_end, y_bar + bar_height], fill='white')
            
            # Добавляем текст
            try:
                font = ImageFont.truetype("arial.ttf", 14)
            except:
                font = ImageFont.load_default()
            draw.text((text_x, y_text), scale_text, fill='white', font=font)
            
            # Конвертируем обратно в numpy array (BGR для OpenCV)
            result = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            return result
        except Exception as e:
            print(f"   ❌ Ошибка добавления шкалы: {e}")
            return image
        



class MacrophageProcessor:
    """Внутренний класс для сегментации и анализа макрофагов."""
    
    def segment_macrophages(nuclei, protein, nuclei_percentile=50, protein_threshold=50, expansion_factor=1.58, 
                       min_macrophage_size=100, max_macrophage_size=3000):
        """Сегментирует макрофаги. Возвращает размеченную маску."""
        # ---- Бинаризация ----
        nuclei_binary = nuclei > np.percentile(nuclei, nuclei_percentile)
        protein_binary = protein > protein_threshold
        seed = nuclei_binary & protein_binary
        seed = morphology.remove_small_objects(seed, min_size=20)

        if not np.any(seed):
            return np.zeros_like(nuclei, dtype=np.int32)

        # ---- Маркируем начальные объекты ----
        labeled_seed = measure.label(seed)
        
        # ---- Расширяем КАЖДОЕ ядро отдельно ----
        expanded_labels = np.zeros_like(labeled_seed, dtype=np.int32)
        
        # Используем константный радиус для расширения (не зависящий от площади ядра!)
        # Или используем фиксированный радиус + небольшое расширение
        FIXED_RADIUS = 30  # ← настройте под ваши изображения (в пикселях)
        
        for label_id in range(1, np.max(labeled_seed) + 1):
            # Маска текущего ядра
            nucleus_mask = (labeled_seed == label_id)
            
            # Находим центроид
            y, x = np.mean(np.where(nucleus_mask)[0]), np.mean(np.where(nucleus_mask)[1])
            
            # Используем ФИКСИРОВАННЫЙ радиус или радиус на основе размера ядра с ограничением
            # Вариант 1: фиксированный радиус (рекомендую начать с этого)
            radius = FIXED_RADIUS
            
            # Вариант 2: радиус на основе размера ядра, но с ограничением
            # area = np.sum(nucleus_mask)
            # radius = min(max(int(np.sqrt(area / np.pi) * 1.5), 20), 50)  # ограничиваем от 20 до 50
            
            # Создаём круг для этого конкретного ядра
            yy, xx = np.ogrid[:nuclei.shape[0], :nuclei.shape[1]]
            circle_mask = (xx - x)**2 + (yy - y)**2 <= radius**2
            
            # Расширяем только область этого ядра
            expanded = circle_mask
            
            # Записываем с уникальной меткой
            expanded_labels[expanded] = label_id
        
        # ---- Удаляем слишком маленькие объекты ----
        expanded_labels = morphology.remove_small_objects(expanded_labels, min_size=min_macrophage_size)
        
        # ---- Дополнительно: удаляем слишком большие объекты ----
        # Это помогает отсечь слипшиеся макрофаги
        from scipy import ndimage
        for label_id in range(1, np.max(expanded_labels) + 1):
            mask = (expanded_labels == label_id)
            size = np.sum(mask)
            if size > max_macrophage_size:
                expanded_labels[mask] = 0
        
        # ---- Перемаркируем ----
        expanded_labels, _ = ndimage.label(expanded_labels > 0)
        
        # ---- Водораздел для разделения слипшихся областей ----
        if np.max(expanded_labels) > 0:
            distance = ndimage.distance_transform_edt(expanded_labels > 0)
            markers = VesicleProcessor.find_peak_markers(distance, expanded_labels > 0, min_distance=5, threshold_rel=0.1)
            if markers.max() == 0:
                markers = measure.label(expanded_labels > 0)
            labels = segmentation.watershed(-distance, markers, mask=expanded_labels > 0)
            return labels
        
        return expanded_labels

    @staticmethod
    def analyze_colocalization(macrophage_labels, vesicles_binary_raw, vesicles_labeled):
        """Анализирует колокализацию макрофагов с везикулами."""
        total_macrophages = np.max(macrophage_labels) if macrophage_labels is not None else 0
        if total_macrophages == 0:
            return {
                'total_macrophages': 0,
                'macrophages_with_vesicles': 0,
                'macrophage_colocalization_percent': 0.0,
                'vesicles_in_macrophages': 0,
                'mean_vesicles_per_macrophage': 0.0
            }

        macrophages_with_vesicles = 0
        vesicle_labels_in_macrophages = set()

        for label in range(1, total_macrophages + 1):
            mask = (macrophage_labels == label)
            if np.any(mask & vesicles_binary_raw):
                macrophages_with_vesicles += 1
                labels_in = np.unique(vesicles_labeled[mask])
                labels_in = labels_in[labels_in > 0]
                vesicle_labels_in_macrophages.update(labels_in)

        vesicles_in_macrophages = len(vesicle_labels_in_macrophages)
        percent = (macrophages_with_vesicles / total_macrophages) * 100 if total_macrophages else 0.0
        mean_vesicles = vesicles_in_macrophages / total_macrophages if total_macrophages else 0.0

        return {
            'total_macrophages': total_macrophages,
            'macrophages_with_vesicles': macrophages_with_vesicles,
            'macrophage_colocalization_percent': percent,
            'vesicles_in_macrophages': vesicles_in_macrophages,
            'mean_vesicles_per_macrophage': mean_vesicles
        }

    @staticmethod
    def draw_macrophages_on_composite(composite, macrophage_labels, color=(255, 165, 0), thickness=2):
        """Рисует оранжевые контуры макрофагов на RGB-композите."""
        if macrophage_labels is None or np.max(macrophage_labels) == 0:
            return composite
        overlay = composite.copy().astype(np.uint8)
        props = measure.regionprops(macrophage_labels)
        for prop in props:
            y, x = prop.centroid
            radius = int(np.sqrt(prop.area / np.pi)) + 2
            # Используем cv2 – он уже импортирован в файле
            cv2.circle(overlay, (int(x), int(y)), radius, color, thickness)
        return overlay