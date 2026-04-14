# multi_lif_analyzer.py
import numpy as np
import cv2
import os
from readlif.reader import LifFile
from skimage import exposure, filters, measure, morphology, segmentation
from scipy import ndimage
import pandas as pd
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
import re
from datetime import datetime
import time
from PIL import Image, ImageDraw, ImageFont

class MultiSampleLifAnalyzer:
    def __init__(self, 
                 # Существующие параметры
                 min_vesicle_size=5, max_vesicle_size=50, 
                 background_method='original', subtraction_factor=0.7,
                 vesicles_contrast_factor=1.0, protein_contrast_factor=1.5,
                 split_large_clusters=True, intensity_ratio_threshold=1.5, 
                 intensity_diff_threshold=20, protein_brightness_factor=1.0,
                 save_igg_data_path=None,
                 
                 # ⭐ НОВЫЕ ПАРАМЕТРЫ
                 protein_subtraction_percent=90,
                 collagen_percentile=90,
                 exclude_patterns=None,
                 channel_background_percentile=70,  # Для apply_colored_channel
                 background_threshold_percentile=85,  # Для find_background_threshold
                 igg_fallback_percentile=90,  # Для get_protein_igg_background
                 default_protein_background=50.0,  # Для get_protein_igg_background
                 min_vesicle_intensity=115,
                 
                 # ⭐ ПАРАМЕТРЫ КОНТРАСТА/ГАММА
                 nuclei_gamma=0.8,
                 vesicles_gamma=0.7,
                 protein_gamma=0.8):
        
        self.gamma_values = {
        0: nuclei_gamma,      # Ядра
        1: 1.0,               # Коллаген (без изменений)
        2: vesicles_gamma,    # Везикулы
        3: protein_gamma      # Белок
    }
    
        # Также добавить другие параметры, которые объявили:
        self.channel_background_percentile = channel_background_percentile
        self.background_threshold_percentile = background_threshold_percentile
        self.igg_fallback_percentile = igg_fallback_percentile
        self.default_protein_background = default_protein_background

        # ⭐ ОСНОВНЫЕ ПАРАМЕТРЫ
        self.min_vesicle_size = min_vesicle_size
        self.max_vesicle_size = max_vesicle_size
        self.background_method = background_method
        self.subtraction_factor = subtraction_factor
        self.vesicles_contrast_factor = vesicles_contrast_factor
        self.protein_contrast_factor = protein_contrast_factor
        self.split_large_clusters = split_large_clusters
        self.protein_brightness_factor = protein_brightness_factor
        
        # ⭐ НОВЫЕ ПАРАМЕТРЫ ДЛЯ РАЗНОСТИ ИНТЕНСИВНОСТЕЙ
        self.intensity_ratio_threshold = intensity_ratio_threshold
        self.intensity_diff_threshold = intensity_diff_threshold
        self.protein_subtraction_percent = protein_subtraction_percent
        self.collagen_percentile = collagen_percentile
        self.min_vesicle_intensity = min_vesicle_intensity

        # ⭐ ПУТЬ ДЛЯ СОХРАНЕНИЯ IgG ДАННЫХ
        self.save_igg_data_path = save_igg_data_path
        
        # ⭐ ИНИЦИАЛИЗАЦИЯ СЛОВАРЕЙ И СПИСКОВ
        self.igg_intensity_data = {}
        self.collagen_intensity_data = {}
        self.protein_intensities_from_igg = []
        if exclude_patterns is None:
            exclude_patterns = ['igg', 'pbs', 'control', 'neg']
        self.exclude_patterns = [p.lower() for p in exclude_patterns]
        # ⭐ ГЛОБАЛЬНОЕ ХРАНИЛИЩЕ IgG ДАННЫХ (для всех анализов)
        self.igg_database = self.load_igg_database() if save_igg_data_path else {}
        
        print(f"      🎛️  Настройки анализатора:")
        print(f"         • Контраст везикул: {vesicles_contrast_factor}")
        print(f"         • Контраст белка: {protein_contrast_factor}")
        print(f"         • Разделение скоплений: {'ВКЛ' if split_large_clusters else 'ВЫКЛ'}")
        print(f"         • Путь для IgG данных: {save_igg_data_path or 'не указан'}")
            
    def extract_sample_info(self, image_name):
        """Извлечение информации о группе и времени из названия изображения"""
        if not image_name:
            return "unknown_group"
        
        image_name_str = str(image_name).lower()
        
        # Определяем группу по времени
        if '30min' in image_name_str or '30 min' in image_name_str:
            time_group = "30min"
        elif '3h' in image_name_str or '3 h' in image_name_str or '3hour' in image_name_str:
            time_group = "3h"
        else:
            time_group = "unknown_time"
        
        # Извлекаем номер мыши
        patterns = [
            r'^(\d+)\s',
            r'\b(\d+)\s*(?:min|h|pbs)\b',
            r'\b(\d+)\s*$',
            r'mouse[_\s]*(\d+)', 
            r'm[_\s]*(\d+)', 
            r'sample[_\s]*(\d+)', 
            r's[_\s]*(\d+)',
            r'img[_\s]*(\d+)', 
            r'image[_\s]*(\d+)',
        ]
        
        mouse_number = "unknown"
        for pattern in patterns:
            match = re.search(pattern, image_name_str)
            if match:
                mouse_number = match.group(1)
                if mouse_number.isdigit():
                    mouse_number = mouse_number.zfill(2)
                    break
        
        return f"{time_group}_mouse_{mouse_number}"
    
    def detect_igg_samples(self, image_name):
        """Определяет, является ли образец IgG по названию"""
        name_lower = str(image_name).lower()
        igg_patterns = ['igg', 'ig g', 'ig-g', 'control']
        return any(pattern in name_lower for pattern in igg_patterns)
    
    def calculate_channel_intensity(self, channel_data):
        """Вычисляет максимальную интенсивность канала"""
        if channel_data is None:
            return 0
        return np.max(channel_data)
    
    def calculate_mean_background(self, channel_data):
        """Вычисляет среднюю интенсивность фона канала"""
        if channel_data is None:
            return 0
        
        # Используем среднее значение интенсивности как фоновый уровень
        mean_intensity = np.max(channel_data)
        print(f"      Средняя максимальная фона: {mean_intensity:.2f}")
        return mean_intensity

    def analyze_vesicle_preservation(self, original_vesicles, corrected_vesicles, sample_name):
        """
        Анализ сохранения везикул после коррекции с учетом минимальной интенсивности
        
        Args:
            original_vesicles: оригинальный канал везикул (до коррекции)
            corrected_vesicles: скорректированный канал везикул (после обработки)
            sample_name: имя образца
        
        Returns:
            bool: True если сохранение хорошее, False если плохое
        """
        if original_vesicles is None or corrected_vesicles is None:
            return False
        
        try:
            print(f"\n   📊 АНАЛИЗ СОХРАНЕНИЯ ВЕЗИКУЛ ДЛЯ: {sample_name}")
            print(f"      🎯 Мин. интенсивность: {self.min_vesicle_intensity}")
            
            # 1. Применяем фильтр минимальной интенсивности к обоим каналам
            # Создаем копии, чтобы не изменять оригиналы
            original_filtered = original_vesicles.copy()
            corrected_filtered = corrected_vesicles.copy()
            
            # Обнуляем пиксели ниже порога интенсивности
            original_filtered[original_filtered < self.min_vesicle_intensity] = 0
            corrected_filtered[corrected_filtered < self.min_vesicle_intensity] = 0
            
            # Статистика до фильтрации
            orig_pixels_before = np.sum(original_vesicles > 0)
            orig_pixels_after = np.sum(original_filtered > 0)
            corr_pixels_before = np.sum(corrected_vesicles > 0)
            corr_pixels_after = np.sum(corrected_filtered > 0)
            
            print(f"      📊 ВЛИЯНИЕ ФИЛЬТРА ИНТЕНСИВНОСТИ:")
            print(f"         • Оригинал: {orig_pixels_before} → {orig_pixels_after} пикселей ({(1-orig_pixels_after/orig_pixels_before)*100:.1f}% отсеяно)")
            print(f"         • Скорректированный: {corr_pixels_before} → {corr_pixels_after} пикселей ({(1-corr_pixels_after/corr_pixels_before)*100:.1f}% отсеяно)")
            
            # 2. Сегментация на оригинальном и скорректированном каналах
            # Используем существующий метод segment_vesicles, который уже учитывает min_vesicle_intensity
            original_binary = self.segment_vesicles(original_filtered)
            corrected_binary = self.segment_vesicles(corrected_filtered)
            
            if original_binary is None or corrected_binary is None:
                print(f"      ⚠️  Не удалось сегментировать везикулы")
                return False
            
            # 3. Подсчет везикул через connected components
            original_labels = measure.label(original_binary)
            corrected_labels = measure.label(corrected_binary)
            
            original_count = np.max(original_labels) if np.max(original_labels) > 0 else 0
            corrected_count = np.max(corrected_labels) if np.max(corrected_labels) > 0 else 0
            
            # 4. Анализ размеров везикул
            original_regions = measure.regionprops(original_labels, intensity_image=original_filtered)
            corrected_regions = measure.regionprops(corrected_labels, intensity_image=corrected_filtered)
            
            if original_regions:
                original_areas = [r.area for r in original_regions]
                original_intensities = [r.mean_intensity for r in original_regions]
                print(f"      📏 Оригинал - размеры: min={min(original_areas):.0f}, avg={np.mean(original_areas):.0f}, max={max(original_areas):.0f}")
                print(f"      💡 Оригинал - интенсивность: min={min(original_intensities):.0f}, avg={np.mean(original_intensities):.0f}, max={max(original_intensities):.0f}")
            
            if corrected_regions:
                corrected_areas = [r.area for r in corrected_regions]
                corrected_intensities = [r.mean_intensity for r in corrected_regions]
                print(f"      📏 Скорректированный - размеры: min={min(corrected_areas):.0f}, avg={np.mean(corrected_areas):.0f}, max={max(corrected_areas):.0f}")
                print(f"      💡 Скорректированный - интенсивность: min={min(corrected_intensities):.0f}, avg={np.mean(corrected_intensities):.0f}, max={max(corrected_intensities):.0f}")
            
            # 5. Вычисляем процент сохранения
            preservation_rate = (corrected_count / original_count * 100) if original_count > 0 else 0
            
            print(f"   📊 СОХРАНЕНИЕ ВЕЗИКУЛ:")
            print(f"      • До коррекции: {original_count}")
            print(f"      • После коррекции: {corrected_count}")
            print(f"      • Сохранено: {preservation_rate:.1f}%")
            
            # 6. Анализ совпадения везикул (пересечение масок)
            if original_binary is not None and corrected_binary is not None:
                # Находим пересечение масок
                intersection = np.logical_and(original_binary > 0, corrected_binary > 0)
                intersection_count = np.sum(intersection > 0)
                
                # Пиксели, которые были в оригинале, но пропали в corrected
                lost_pixels = np.sum(np.logical_and(original_binary > 0, corrected_binary == 0))
                
                print(f"      • Пересечение масок: {intersection_count} пикселей")
                print(f"      • Потеряно пикселей: {lost_pixels} ({lost_pixels/np.sum(original_binary>0)*100:.1f}%)")
            
            # 7. Оценка качества сохранения
            if preservation_rate < 50:  # Если потеряли более 50% везикул
                print(f"      🚨 КРИТИЧНО: Потеряно более 50% везикул!")
                print(f"      💡 Рекомендация: уменьшить min_vesicle_intensity или vesicle_subtraction_factor")
                return False
            elif preservation_rate < 70:  # Если потеряли 30-50%
                print(f"      ⚠️  ВНИМАНИЕ: Потеряно {100-preservation_rate:.1f}% везикул")
                print(f"      💡 Рекомендация: проверить параметры коррекции")
                return True
            elif preservation_rate > 95:  # Если сохранили почти все
                print(f"      ✅ Отличное сохранение везикул!")
                return True
            else:
                print(f"      👍 Хорошее сохранение везикул")
                return True
            
        except Exception as e:
            print(f"   ❌ Ошибка анализа сохранения везикул: {e}")
            import traceback
            traceback.print_exc()
            return False
        

    def subtract_background_intensity(self, channel_data, background_intensity, method='conservative'):
        """Улучшенное вычитание фона с разными стратегиями"""
        if channel_data is None:
            return None
        
        try:
            max_intensity = np.max(channel_data)
            mean_intensity = np.mean(channel_data)
            
            print(f"      🧮 До вычитания: max={max_intensity:.1f}, mean={mean_intensity:.1f}, background={background_intensity:.1f}")
            
            if method == 'conservative':
                # КОНСЕРВАТИВНЫЙ МЕТОД: вычитаем только часть фона
                subtraction_factor = 0.7  # Вычитаем только 70% фона
                amount_to_subtract = background_intensity * subtraction_factor
                
                # Защита от слишком агрессивного вычитания
                if amount_to_subtract > mean_intensity * 0.9:
                    amount_to_subtract = mean_intensity * 0.7
                    print(f"      ⚠️  СЛИШКОМ МНОГО! Снижено до: {amount_to_subtract:.1f}")
                
                result = channel_data.astype(np.float32) - amount_to_subtract
                result = np.clip(result, 0, 255).astype(np.uint8)
                
                after_max = np.max(result)
                after_mean = np.mean(result)
                print(f"      ✅ После вычитания: max={after_max:.1f}, mean={after_mean:.1f}")
                print(f"      📉 Вычтено: {amount_to_subtract:.1f} ({subtraction_factor*100:.0f}% фона)")
                
            elif method == 'adaptive':
                # АДАПТИВНЫЙ МЕТОД: вычитаем пропорционально интенсивности
                result = channel_data.astype(np.float32)
                
                # Вычитаем только из пикселей выше определенного порога
                threshold = background_intensity * 1.2
                mask = result > threshold
                result[mask] = result[mask] - background_intensity * 0.3
                result = np.clip(result, 0, 255).astype(np.uint8)
                
                print(f"      🔄 Адаптивное вычитание применено")
                
            else:
                # ОРИГИНАЛЬНЫЙ МЕТОД (полное вычитание)
                result = channel_data.astype(np.float32) - background_intensity
                result = np.clip(result, 0, 255).astype(np.uint8)
                print(f"      ⚠️  Использовано полное вычитание фона")
            
            return result
            
        except Exception as e:
            print(f"      ❌ Ошибка вычитания фона: {e}")
            return channel_data
    
    def add_scale_bar(self, image, scale_length_pixels=100, scale_text="100 μm", 
                 bar_height=6, position='bottom_right', margin=25):
        """Добавляет белую шкалу с текстом внизу (без черного фона)"""
        try:
            if image is None:
                return None
                
            # Конвертируем в PIL Image если это numpy array
            if isinstance(image, np.ndarray):
                if len(image.shape) == 3 and image.shape[2] == 3:
                    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                else:
                    # Если одноканальное, конвертируем в RGB
                    pil_image = Image.fromarray(image).convert('RGB')
            else:
                pil_image = image.copy()
            
            draw = ImageDraw.Draw(pil_image)
            width, height = pil_image.size
            
            # Определяем положение шкалы (текст ПОД шкалой)
            if position == 'bottom_right':
                x_start = width - scale_length_pixels - margin
                x_end = width - margin
                y_bar = height - margin - bar_height
                y_text = y_bar + bar_height + 5  # Текст под шкалой
                text_x = x_start + (scale_length_pixels - 40) // 2  # Центрируем текст
            else:  # bottom_center
                x_start = (width - scale_length_pixels) // 2
                x_end = x_start + scale_length_pixels
                y_bar = height - margin - bar_height
                y_text = y_bar + bar_height + 5
                text_x = x_start + (scale_length_pixels - 40) // 2
            
            # Рисуем белую шкалу (убрали черную обводку)
            draw.rectangle([x_start, y_bar, x_end, y_bar + bar_height], fill='white')
            
            # Добавляем текст под шкалой (без черного фона)
            try:
                font_sizes = [14, 12, 10, 8]
                font = None
                for size in font_sizes:
                    try:
                        font = ImageFont.truetype("arial.ttf", size)
                        break
                    except:
                        try:
                            font = ImageFont.truetype("Arial.ttf", size)
                            break
                        except:
                            continue
                
                if font is None:
                    font = ImageFont.load_default()
                
                # Белый текст без фона
                draw.text((text_x, y_text), scale_text, fill='white', font=font)
                
            except Exception as e:
                print(f"   ⚠️ Ошибка добавления текста шкалы: {e}")
                # Простой текст без шрифта
                draw.text((text_x, y_text), scale_text, fill='white')
            
            # Конвертируем обратно в numpy array
            result = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            return result
            
        except Exception as e:
            print(f"   ❌ Ошибка добавления шкалы: {e}")
            return image
        

    def enhance_contrast(self, channel_data, contrast_factor=1.0):
        """Увеличивает контраст на указанный процент"""
        if channel_data is None:
            return None
        
        if contrast_factor == 1.0:
            # Если коэффициент = 1.0, возвращаем исходное изображение без изменений
            return channel_data
        
        print(f"      🔧 Применение контраста: коэффициент = {contrast_factor}")
        
        try:
            # Преобразуем в float для расчетов
            channel_float = channel_data.astype(np.float32)
            
            # Нормализуем к диапазону 0-1
            channel_normalized = channel_float / 255.0
            
            # Применяем gamma коррекцию для увеличения контраста
            gamma = 1.0 / contrast_factor
            channel_enhanced = np.power(channel_normalized, gamma)
            
            # Возвращаем к диапазону 0-255
            channel_enhanced = (channel_enhanced * 255).astype(np.uint8)
            
            # Диагностика
            orig_mean = np.mean(channel_data)
            enh_mean = np.mean(channel_enhanced)
            print(f"      ✅ Контраст применен: средняя яркость {orig_mean:.1f} → {enh_mean:.1f}")
            
            return channel_enhanced
            
        except Exception as e:
            print(f"      ❌ Ошибка увеличения контраста: {e}")
            return channel_data
    

    def find_background_threshold(self, channel_data, min_pixel_count=5):
        """
        Улучшенный метод нахождения порога фона - более консервативный
        """
        if channel_data is None:
            return 0
        
        try:
            # Создаем гистограмму
            hist, bin_edges = np.histogram(channel_data.flatten(), bins=256, range=(0, 255))
            
            # Ищем точку, где гистограмма "касается" оси X
            threshold = 0
            found_threshold = False
            
            # Проходим по гистограмме от высоких интенсивностей к низким
            for i in range(255, 0, -1):
                if hist[i] <= min_pixel_count:
                    # Проверяем следующие несколько бинов чтобы убедиться что это не случайный выброс
                    consecutive_low = True
                    check_range = min(5, i)  # Уменьшили до 5 для большей консервативности
                    
                    for j in range(1, check_range):
                        if i - j >= 0 and hist[i - j] > min_pixel_count * 5:  # Более либеральный порог
                            consecutive_low = False
                            break
                    
                    if consecutive_low:
                        threshold = bin_edges[i]
                        found_threshold = True
                        break
            
            # Если не нашли явный порог, используем более консервативный перцентиль
            if not found_threshold:
                threshold = np.percentile(channel_data, self.background_threshold_percentile)  # Снизили с 95% до 85%
                print(f"      ⚠️  Используется перцентиль 85%: {threshold:.2f}")
            
            # ДИАГНОСТИКА: сравниваем с другими перцентилями
            p50 = np.percentile(channel_data, 50)
            p75 = np.percentile(channel_data, 75)
            p90 = np.percentile(channel_data, 90)
            max_intensity = np.max(channel_data)
            mean_intensity = np.mean(channel_data)
            
            print(f"      📊 Интенсивности: max={max_intensity:.1f}, mean={mean_intensity:.1f}")
            print(f"      📈 Перцентили: 50%={p50:.1f}, 75%={p75:.1f}, 90%={p90:.1f}")
            print(f"      🎯 Выбранный порог: {threshold:.1f}")
            
            # ЗАЩИТА: не позволяем порогу быть слишком высоким
            if threshold > max_intensity * 0.8:  # Если порог > 80% от максимума
                threshold = max_intensity * 0.6  # Снижаем до 60%
                print(f"      ⚠️  СЛИШКОМ ВЫСОКИЙ ПОРОГ! Установлен на {threshold:.1f}")
            
            return threshold
            
        except Exception as e:
            print(f"      ❌ Ошибка анализа гистограммы: {e}")
            return np.percentile(channel_data, 50)  # Возвращаем медиану в случае ошибки
        
    def load_igg_database(self):
        """Загружает базу данных IgG интенсивностей из файла"""
        if not self.save_igg_data_path:
            return {}
        
        database_file = os.path.join(self.save_igg_data_path, "igg_protein_intensities.json")
        
        if os.path.exists(database_file):
            try:
                import json
                with open(database_file, 'r') as f:
                    database = json.load(f)
                print(f"      📖 Загружена база IgG данных из: {database_file}")
                print(f"      📊 Данных: {len(database.get('protein_intensities', []))} образцов")
                return database
            except Exception as e:
                print(f"      ⚠️  Ошибка загрузки базы IgG: {e}")
                return {}
        else:
            print(f"      📝 База IgG данных не найдена, создается новая")
            return {
                'protein_intensities': [],
                'protein_igg_background': None,
                'samples_count': 0,
                'last_update': None
            }

    def save_igg_database(self):
        """Сохраняет базу данных IgG интенсивностей в файл"""
        if not self.save_igg_data_path:
            return False
        
        try:
            import json
            from datetime import datetime
            
            database_file = os.path.join(self.save_igg_data_path, "igg_protein_intensities.json")
            
            # Подготавливаем данные для сохранения
            database_data = {
                'protein_intensities': self.igg_database.get('protein_intensities', []),
                'protein_igg_background': self.igg_database.get('protein_igg_background'),
                'samples_count': len(self.igg_database.get('protein_intensities', [])),
                'last_update': datetime.now().isoformat(),
                'analysis_parameters': {
                    'percentile': 90,  # ⭐ УКАЗЫВАЕМ КАКОЙ ПЕРЦЕНТИЛЬ ИСПОЛЬЗУЕМ
                    'description': 'Intensity values from IgG control samples'
                }
            }
            
            # Создаем папку если не существует
            os.makedirs(os.path.dirname(database_file), exist_ok=True)
            
            # Сохраняем в JSON
            with open(database_file, 'w') as f:
                json.dump(database_data, f, indent=4)
            
            print(f"      💾 База IgG данных сохранена в: {database_file}")
            print(f"      📊 Сохранено интенсивностей: {len(database_data['protein_intensities'])}")
            
            return True
        except Exception as e:
            print(f"      ❌ Ошибка сохранения базы IgG: {e}")
            return False

    def update_igg_database(self, new_intensities):
        """Обновляет базу данных IgG новыми интенсивностями"""
        if not hasattr(self, 'igg_database'):
            self.igg_database = {'protein_intensities': []}
        
        # Добавляем новые интенсивности
        current_intensities = self.igg_database.get('protein_intensities', [])
        current_intensities.extend(new_intensities)
        
        # Убираем дубликаты и сортируем
        unique_intensities = sorted(list(set(current_intensities)))
        self.igg_database['protein_intensities'] = unique_intensities
        
        # Пересчитываем фон
        if unique_intensities:
            self.igg_database['protein_igg_background'] = np.percentile(unique_intensities, 90)
            self.igg_database['samples_count'] = len(unique_intensities)
            
            print(f"      🔄 База IgG обновлена:")
            print(f"         • Всего интенсивностей: {len(unique_intensities)}")
            print(f"         • 90% перцентиль: {self.igg_database['protein_igg_background']:.2f}")
            print(f"         • Диапазон: {min(unique_intensities):.1f} - {max(unique_intensities):.1f}")
        
        # Сохраняем обновленную базу
        self.save_igg_database()
        
        return True

    def get_protein_igg_background(self):
        """Получает IgG фон для белка с резервной стратегией"""
        # 1. Пробуем использовать текущие данные анализа
        if hasattr(self, 'protein_igg_background') and self.protein_igg_background > 0:
            print(f"      ✅ Используем текущий IgG фон: {self.protein_igg_background:.2f}")
            return self.protein_igg_background
        
        # 2. Пробуем использовать базу данных
        elif hasattr(self, 'igg_database') and self.igg_database.get('protein_igg_background'):
            background = self.igg_database['protein_igg_background']
            samples_count = self.igg_database.get('samples_count', 0)
            print(f"      ✅ Используем IgG фон из базы данных: {background:.2f}")
            print(f"         (на основе {samples_count} исторических образцов)")
            return background
        
        # 3. Если есть какие-то интенсивности, вычисляем из них
        elif hasattr(self, 'protein_intensities_from_igg') and self.protein_intensities_from_igg:
            # ⭐ ИСПРАВЛЕНИЕ: Используем параметр из конструктора вместо фиксированного 90
            fallback_percentile = getattr(self, 'igg_fallback_percentile', 90)  # По умолчанию 90 если не задан
            background = np.percentile(self.protein_intensities_from_igg, fallback_percentile)
            print(f"      ⚠️  Используем IgG фон из текущих образцов: {background:.2f}")
            print(f"         (только {len(self.protein_intensities_from_igg)} образцов, {fallback_percentile}% перцентиль)")
            return background
        
        # 4. Резервный вариант: используем дефолтное значение
        else:
            # ⭐ ИСПРАВЛЕНИЕ: Используем параметр из конструктора вместо фиксированного 50.0
            default_background = getattr(self, 'default_protein_background', 50.0)  # По умолчанию 50.0 если не задан
            print(f"      ⚠️  IgG фон не найден, используется дефолтное значение: {default_background:.2f}")
            return default_background
    
    def load_igg_from_file(self, igg_lif_file_path, output_base_dir):
        """Загружает ВСЕ образцы из IgG файла как контроли и сохраняет изображения"""
        print(f"📖 Загрузка ВСЕХ образцов из IgG файла: {igg_lif_file_path}")
        
        if not os.path.exists(igg_lif_file_path):
            print(f"❌ IgG файл не найден: {igg_lif_file_path}")
            print(f"   ⚠️  Продолжаем без отдельного IgG файла")
            return False
        
        try:
            # Создаем папку для IgG образцов
            base_name = os.path.splitext(os.path.basename(igg_lif_file_path))[0]
            igg_output_dir = os.path.join(output_base_dir, f"igg_controls_{base_name}")
            os.makedirs(igg_output_dir, exist_ok=True)
            print(f"📁 Папка для IgG образцов: {igg_output_dir}")
            
            lif_file = LifFile(igg_lif_file_path)
            total_images = len(lif_file.image_list)
            print(f"✅ Найдено {total_images} изображений в IgG файле")
            
            igg_samples_found = 0
            all_protein_intensities = []
            
            for i, image_item in enumerate(lif_file.image_list):
                if isinstance(image_item, dict):
                    image_name = image_item.get('name', f'image_{i+1}')
                else:
                    image_name = getattr(image_item, 'name', f'image_{i+1}')
                
                print(f"   🔍 Загрузка образца {i+1}/{total_images}: {image_name}")
                
                # Используем существующий метод force_load_all_channels
                sample_channels = self.force_load_all_channels(image_item, igg_lif_file_path)
                
                if sample_channels:
                    igg_samples_found += 1
                    print(f"   ✅ Образец загружен: {image_name}")
                    
                    # Сохраняем IgG образцы как основные образцы
                    self.save_igg_sample_images(sample_channels, image_name, igg_output_dir)
                    
                    # Извлекаем интенсивность белка (канал 3)
                    if 3 in sample_channels:
                        protein_channel = sample_channels[3]
                        max_intensity = float(np.max(protein_channel))
                        all_protein_intensities.append(max_intensity)
                        
                        # Диагностика интенсивности
                        mean_intensity = float(np.mean(protein_channel))
                        std_intensity = float(np.std(protein_channel))
                        
                        print(f"      🎯 Канал белка (4):")
                        print(f"         • Максимальная интенсивность = {max_intensity:.2f}")
                        print(f"         • Средняя интенсивность = {mean_intensity:.2f}")
                        print(f"         • Стандартное отклонение = {std_intensity:.2f}")
                    
                    # Также сохраняем интенсивности других каналов для диагностики
                    for channel_idx, channel_data in sample_channels.items():
                        channel_max = float(np.max(channel_data))
                        channel_mean = float(np.mean(channel_data))
                        
                        channel_names = {
                            0: "Ядра",
                            1: "Коллаген", 
                            2: "Везикулы",
                            3: "Белок"
                        }
                        
                        channel_name = channel_names.get(channel_idx, f"Канал {channel_idx}")
                        print(f"      📊 {channel_name}: max={channel_max:.2f}, mean={channel_mean:.2f}")
            
            # ⭐ ОБНОВЛЯЕМ БАЗУ ДАННЫХ
            if all_protein_intensities:
                print(f"\n      🔄 Обновление базы IgG данных...")
                
                # Статистика перед обновлением
                print(f"      📈 Статистика интенсивностей белка:")
                print(f"         • Количество образцов: {len(all_protein_intensities)}")
                print(f"         • Минимальная: {min(all_protein_intensities):.2f}")
                print(f"         • Максимальная: {max(all_protein_intensities):.2f}")
                print(f"         • Средняя: {np.mean(all_protein_intensities):.2f}")
                print(f"         • Медиана: {np.median(all_protein_intensities):.2f}")
                
                # Вычисляем перцентили для диагностики
                percentiles = [50, 75, 90, 95, 99]
                for p in percentiles:
                    value = np.percentile(all_protein_intensities, p)
                    print(f"         • {p}% перцентиль: {value:.2f}")
                
                # Обновляем базу данных
                self.update_igg_database(all_protein_intensities)
                
                # Сохраняем в текущий анализ
                self.protein_intensities_from_igg = all_protein_intensities
                
                # Вычисляем 90% перцентиль для текущего анализа
                self.protein_igg_background = float(np.percentile(all_protein_intensities, 90))
                
                print(f"\n      📊 РЕЗУЛЬТАТЫ АНАЛИЗА IgG:")
                print(f"         • Загружено IgG образцов: {igg_samples_found}/{total_images}")
                print(f"         • IgG фон для белка (90% перцентиль): {self.protein_igg_background:.2f}")
                print(f"         • Будет вычтено из целевых образцов: {self.protein_igg_background * 0.9:.2f} (90%)")
                print(f"         • Данные сохранены для будущих анализов")
                
                # Сохраняем дополнительную статистику в файл
                stats_file = os.path.join(igg_output_dir, "igg_statistics.txt")
                with open(stats_file, 'w') as f:
                    f.write("СТАТИСТИКА IgG ОБРАЗЦОВ\n")
                    f.write("=" * 50 + "\n")
                    f.write(f"Файл: {igg_lif_file_path}\n")
                    f.write(f"Дата анализа: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Всего образцов: {total_images}\n")
                    f.write(f"Успешно загружено: {igg_samples_found}\n\n")
                    
                    f.write("ИНТЕНСИВНОСТИ БЕЛКА (канал 4):\n")
                    f.write(f"• IgG фон (90% перцентиль): {self.protein_igg_background:.2f}\n")
                    f.write(f"• Минимальная: {min(all_protein_intensities):.2f}\n")
                    f.write(f"• Максимальная: {max(all_protein_intensities):.2f}\n")
                    f.write(f"• Средняя: {np.mean(all_protein_intensities):.2f}\n")
                    f.write(f"• Медиана: {np.median(all_protein_intensities):.2f}\n\n")
                    
                    f.write("ПЕРЦЕНТИЛИ:\n")
                    for p in percentiles:
                        value = np.percentile(all_protein_intensities, p)
                        f.write(f"• {p}%: {value:.2f}\n")
                    
                    f.write(f"\nИНТЕНСИВНОСТИ ПО ОБРАЗЦАМ:\n")
                    for i, intensity in enumerate(all_protein_intensities):
                        f.write(f"Образец {i+1}: {intensity:.2f}\n")
                
                print(f"      📄 Подробная статистика сохранена в: {stats_file}")
            
            else:
                print(f"      ⚠️  Не удалось извлечь интенсивности белка из IgG образцов")
                print(f"      ℹ️  Проверьте, что в файле есть канал 4 (белок)")
            
            if igg_samples_found > 0:
                print(f"✅ УСПЕХ: Загружены IgG данные из {igg_samples_found} образцов")
                print(f"📁 IgG изображения сохранены в: {igg_output_dir}")
                return True
            else:
                print("❌ ОШИБКА: Не удалось загрузить ни одного образца из IgG файла")
                return False
                
        except Exception as e:
            print(f"❌ Ошибка загрузки IgG файла: {e}")
            import traceback
            traceback.print_exc()
            print(f"   ⚠️  Продолжаем без отдельного IgG файла")
            return False
    
    def find_igg_samples_in_main_file(self, samples_data):
        """Находит IgG образцы в основном файле и добавляет их в IgG данные"""
        print("🔍 Поиск IgG образцов в основном файле...")
        
        all_protein_intensities = []
        
        for mouse_id, samples in samples_data.items():
            for sample in samples:
                if self.detect_igg_samples(sample['sample_name']):
                    print(f"   ✅ Найден IgG образец: {sample['sample_name']}")
                    channels = sample['channels']
                    
                    if 3 in channels:
                        protein_channel = channels[3]
                        max_intensity = np.max(protein_channel)
                        all_protein_intensities.append(float(max_intensity))
                        print(f"      🎯 Канал белка: интенсивность = {max_intensity:.2f}")
        
        # Обновляем IgG фон для белка с учетом образцов из основного файла
        if all_protein_intensities:
                print(f"\n   🔄 Обновление базы IgG данных из основного файла...")
                self.update_igg_database(all_protein_intensities)
                
                # Объединяем с существующими данными
                if hasattr(self, 'protein_intensities_from_igg'):
                    combined_intensities = self.protein_intensities_from_igg + all_protein_intensities
                else:
                    combined_intensities = all_protein_intensities
                
                self.protein_intensities_from_igg = combined_intensities
                self.protein_igg_background = np.percentile(combined_intensities, 90)
                
                print(f"   📊 Объединенный IgG фон: {self.protein_igg_background:.2f}")
                print(f"   📈 Всего интенсивностей: {len(combined_intensities)}")
            
        return len(all_protein_intensities) > 0
    
    def save_igg_sample_images(self, sample_channels, sample_name, output_dir):
        """Сохраняет IgG образцы с цветами как основные образцы"""
        try:
            print(f"   🎨 Сохранение IgG образца: {sample_name}")
            
            # Предобработка каналов
            nuclei_channel = self.preprocess_channel(sample_channels.get(0), 0) if 0 in sample_channels else None
            collagen_channel = self.preprocess_channel(sample_channels.get(1), 1) if 1 in sample_channels else None
            vesicles_channel = self.preprocess_channel(sample_channels.get(2), 2) if 2 in sample_channels else None
            protein_channel = self.preprocess_channel(sample_channels.get(3), 3) if 3 in sample_channels else None
            
            # Создаем композитное изображение
            if nuclei_channel is not None and vesicles_channel is not None and protein_channel is not None:
                height, width = nuclei_channel.shape
                composite = np.zeros((height, width, 3), dtype=np.uint8)
                
                composite[:,:,0] = nuclei_channel    # Синий - Ядра
                composite[:,:,1] = protein_channel   # Зеленый - Белок
                composite[:,:,2] = vesicles_channel  # Красный - Везикулы
                
                # Добавляем шкалу
                composite_with_scale = self.add_scale_bar(
                    composite, 
                    scale_length_pixels=100, 
                    scale_text="100 μm",
                    position='bottom_right',
                    margin=20
                )
                
                if composite_with_scale is not None:
                    composite = composite_with_scale
            
            # Сохраняем композит
            safe_name = "".join(c for c in sample_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            composite_filename = f"{safe_name}_composite.png"
            composite_path = os.path.join(output_dir, composite_filename)
            
            success = cv2.imwrite(composite_path, composite)
            if success:
                print(f"   ✅ IgG композит сохранен: {composite_filename}")
            
            # Сохраняем отдельные каналы
            if nuclei_channel is not None:
                nuclei_colored = self.apply_colored_channel(nuclei_channel, 'blue')
                nuclei_colored = self.add_scale_bar(nuclei_colored, scale_length_pixels=100, scale_text="100 μm")
                cv2.imwrite(os.path.join(output_dir, f"{safe_name}_channel0_nuclei.png"), nuclei_colored)
            
            if collagen_channel is not None:
                collagen_colored = self.apply_colored_channel(collagen_channel, 'green')
                collagen_colored = self.add_scale_bar(collagen_colored, scale_length_pixels=100, scale_text="100 μm")
                cv2.imwrite(os.path.join(output_dir, f"{safe_name}_channel1_collagen.png"), collagen_colored)
            
            if vesicles_channel is not None:
                vesicles_colored = self.apply_colored_channel(vesicles_channel, 'red')
                vesicles_colored = self.add_scale_bar(vesicles_colored, scale_length_pixels=100, scale_text="100 μm")
                cv2.imwrite(os.path.join(output_dir, f"{safe_name}_channel2_vesicles.png"), vesicles_colored)
            
            if protein_channel is not None:
                protein_colored = self.apply_colored_channel(protein_channel, 'green')
                protein_colored = self.add_scale_bar(protein_colored, scale_length_pixels=100, scale_text="100 μm")
                cv2.imwrite(os.path.join(output_dir, f"{safe_name}_channel3_protein.png"), protein_colored)
                
        except Exception as e:
            print(f"   ❌ Ошибка сохранения IgG образца: {e}")


    def pil_to_numpy(self, pil_image):
        """Конвертирует PIL Image в numpy array"""
        if pil_image.mode == 'I;16':
            array = np.array(pil_image, dtype=np.uint16)
        elif pil_image.mode == 'I':
            array = np.array(pil_image, dtype=np.int32)
        elif pil_image.mode == 'F':
            array = np.array(pil_image, dtype=np.float32)
        else:
            array = np.array(pil_image)
        
        return array

    def force_load_all_channels(self, image_dict, lif_file_path):
        """Принудительная загрузка ВСЕХ каналов"""
        print(f"   🔍 Принудительная загрузка каналов...")
        
        sample_channels = {}
        
        try:
            # Получаем информацию о каналах из словаря
            num_channels = image_dict.get('channels', 4)
            dims = image_dict.get('dims', None)
            image_name = image_dict.get('name', 'unknown')
            
            print(f"   📐 Размеры: {dims}")
            print(f"   📊 Каналов: {num_channels}")
            print(f"   🏷️ Имя изображения: {image_name}")
            
            # Загружаем LIF файл
            lif_file = LifFile(lif_file_path)
            
            # Получаем изображение по индексу напрямую
            image_index = None
            for i, img_item in enumerate(lif_file.image_list):
                img_name = img_item.get('name', '') if isinstance(img_item, dict) else getattr(img_item, 'name', '')
                if img_name == image_name:
                    image_index = i
                    break
            
            if image_index is None:
                print(f"   ❌ Не найден индекс изображения для: {image_name}")
                return {}
            
            print(f"   ✅ Найден индекс изображения: {image_index}")
            
            # Получаем объект LifImage
            lif_image = lif_file.get_image(image_index)
            
            # Загружаем все каналы
            for channel_idx in range(num_channels):
                try:
                    print(f"   🔍 Загрузка канала {channel_idx}...")
                    
                    channel_loaded = False
                    
                    # Метод 1: используем get_frame с параметрами канала
                    try:
                        channel_data = lif_image.get_frame(c=channel_idx)
                        
                        if channel_data is not None:
                            if isinstance(channel_data, Image.Image):
                                print(f"   ✅ Канал {channel_idx} загружен как PIL Image")
                                
                                # Конвертируем PIL в numpy
                                channel_array = self.pil_to_numpy(channel_data)
                                
                                # Нормализуем в 8-bit если нужно
                                if channel_array.dtype != np.uint8:
                                    if channel_array.dtype == np.uint16:
                                        channel_array = (channel_array / 256).astype(np.uint8)
                                    else:
                                        channel_array = cv2.normalize(
                                            channel_array, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
                                        )
                                
                                sample_channels[channel_idx] = channel_array
                                channel_loaded = True
                            else:
                                print(f"   ✅ Канал {channel_idx} загружен как numpy array")
                                
                                if channel_data.dtype != np.uint8:
                                    channel_data = cv2.normalize(
                                        channel_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
                                    )
                                
                                sample_channels[channel_idx] = channel_data
                                channel_loaded = True
                                
                    except Exception as e:
                        print(f"   ❌ Ошибка загрузки канала {channel_idx} (метод 1): {e}")
                    
                    if not channel_loaded:
                        # Метод 2: пробуем загрузить все каналы и извлечь нужный
                        try:
                            all_channels = lif_image.get_frame()
                            
                            if all_channels is not None:
                                if isinstance(all_channels, Image.Image):
                                    all_array = self.pil_to_numpy(all_channels)
                                    
                                    if len(all_array.shape) == 3 and all_array.shape[2] >= num_channels:
                                        channel_array = all_array[:, :, channel_idx]
                                        print(f"   ✅ Канал {channel_idx} извлечен из 3D массива")
                                        
                                        if channel_array.dtype != np.uint8:
                                            channel_array = cv2.normalize(
                                                channel_array, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
                                            )
                                        
                                        sample_channels[channel_idx] = channel_array
                                        channel_loaded = True
                                    elif len(all_array.shape) == 2 and channel_idx == 0:
                                        if all_array.dtype != np.uint8:
                                            all_array = cv2.normalize(
                                                all_array, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
                                            )
                                        
                                        sample_channels[channel_idx] = all_array
                                        channel_loaded = True
                                else:
                                    if isinstance(all_channels, np.ndarray):
                                        if len(all_channels.shape) == 3 and all_channels.shape[2] > channel_idx:
                                            channel_array = all_channels[:, :, channel_idx]
                                            print(f"   ✅ Канал {channel_idx} извлечен из 3D numpy массива")
                                            
                                            if channel_array.dtype != np.uint8:
                                                channel_array = cv2.normalize(
                                                    channel_array, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
                                                )
                                            
                                            sample_channels[channel_idx] = channel_array
                                            channel_loaded = True
                                        elif len(all_channels.shape) == 2 and channel_idx == 0:
                                            channel_array = all_channels
                                            print(f"   ✅ Единственный канал извлечен из numpy")
                                            
                                            if channel_array.dtype != np.uint8:
                                                channel_array = cv2.normalize(
                                                    channel_array, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
                                                )
                                            
                                            sample_channels[channel_idx] = channel_array
                                            channel_loaded = True
                        except Exception as e:
                            print(f"   ❌ Ошибка загрузки канала {channel_idx} (метод 2): {e}")
                    
                    if not channel_loaded:
                        print(f"   ❌ Не удалось загрузить канал {channel_idx}")
                            
                except Exception as e:
                    print(f"   ❌ Критическая ошибка обработки канала {channel_idx}: {e}")
                    continue
                    
            print(f"   📊 Успешно загружено каналов: {len(sample_channels)}")
            return sample_channels
            
        except Exception as e:
            print(f"   ❌ Критическая ошибка загрузки каналов: {e}")
            return {}

    def load_all_samples_from_lif(self, lif_file_path):
        """Загрузка всех образцов из LIF файла БЕЗ коррекции IgG"""
        print(f"📖 Загрузка основного LIF файла: {lif_file_path}")
        
        try:
            lif_file = LifFile(lif_file_path)
            total_images = len(lif_file.image_list)
            print(f"✅ Найдено {total_images} изображений в основном LIF файле")
            
        except Exception as e:
            print(f"❌ Ошибка загрузки основного LIF файла: {e}")
            return {}
        
        samples_data = {}
        
        # Загрузка всех образцов БЕЗ коррекции IgG
        print(f"\n🔍 Загрузка образцов БЕЗ коррекции IgG...")
        
        for i, image_item in enumerate(lif_file.image_list):
            print(f"\n🎯 Обработка изображения {i+1}/{total_images}:")
            
            if isinstance(image_item, dict):
                image_name = image_item.get('name', f'image_{i+1}')
            else:
                image_name = getattr(image_item, 'name', f'image_{i+1}')
            
            print(f"   Название: {image_name}")
            
            mouse_id = self.extract_sample_info(image_name)
            print(f"   Группа: {mouse_id}")
            
            sample_channels = self.force_load_all_channels(image_item, lif_file_path)
            
            if not sample_channels:
                print("   ⚠️ Не удалось загрузить каналы, пропускаем")
                continue
            
            if mouse_id not in samples_data:
                samples_data[mouse_id] = []
            
            sample_info = {
                'sample_name': image_name,
                'channels': sample_channels,
                'mouse_id': mouse_id,
                'channels_count': len(sample_channels),
                'original_index': i,
                'is_igg': self.detect_igg_samples(image_name)
            }
            
            samples_data[mouse_id].append(sample_info)
            print(f"   💾 Образец сохранен в группу: {mouse_id}")
        
        # ПОИСК IgG ОБРАЗЦОВ В ОСНОВНОМ ФАЙЛЕ
        self.find_igg_samples_in_main_file(samples_data)
        
        # Выводим статистику
        print(f"\n📊 СТАТИСТИКА ГРУППИРОВКИ:")
        total_samples = 0
        for mouse_id, samples in samples_data.items():
            samples_in_group = len(samples)
            total_samples += samples_in_group
            igg_count = sum(1 for s in samples if s['is_igg'])
            print(f"   {mouse_id}: {samples_in_group} образцов ({igg_count} IgG)")
        
        print(f"   ВСЕГО: {total_samples} образцов")
        return samples_data

    
    def preprocess_channel(self, img, channel_type):
        """Предобработка канала"""
        if img is None:
            return None
            
        try:
            # Уменьшение шума
            img_denoised = cv2.medianBlur(img, 3)
            
            # Уменьшение фона
            try:
                threshold = filters.threshold_otsu(img_denoised)
                background_reduced = cv2.subtract(img_denoised, int(threshold * 0.7))
                background_reduced = np.clip(background_reduced, 0, 255)
            except:
                background_reduced = img_denoised
            
            # Увеличение контраста
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img_contrast = clahe.apply(background_reduced)
            
            # Gamma коррекция
            gamma = self.gamma_values.get(channel_type, 0.8)
            img_enhanced = exposure.adjust_gamma(img_contrast, gamma=gamma)
            
            return img_enhanced
            
        except Exception as e:
            print(f"   ❌ Ошибка предобработки канала {channel_type}: {e}")
            return None

    def apply_colored_channel(self, gray_channel, color, keep_background_black=True):
        if gray_channel is None:
            return None
            
        height, width = gray_channel.shape
        colored = np.zeros((height, width, 3), dtype=np.uint8)
        
        if keep_background_black:
            # ⭐ УЛУЧШЕННЫЙ ПОРОГ ДЛЯ УДАЛЕНИЯ ФОНА
            # Используем адаптивный порог на основе статистики
            if np.max(gray_channel) > 0:
                # Порог = 30% от максимальной интенсивности или фиксированный минимум
                background_threshold = max(15, np.percentile(gray_channel, self.channel_background_percentile))
            else:
                background_threshold = 15
                
            # Создаем маску ТОЛЬКО для достаточно ярких пикселей
            mask = gray_channel > background_threshold
            
            print(f"      🎯 Адаптивный порог фона: {background_threshold}")
            print(f"      📊 Ярких пикселей: {np.sum(mask)}/{gray_channel.size} ({np.sum(mask)/gray_channel.size*100:.1f}%)")
            
            if color == 'red':
                colored[:,:,2][mask] = gray_channel[mask]
            elif color == 'green':
                colored[:,:,1][mask] = gray_channel[mask]
            elif color == 'blue':
                colored[:,:,0][mask] = gray_channel[mask]
            elif color == 'yellow':
                colored[:,:,1][mask] = gray_channel[mask]
                colored[:,:,2][mask] = gray_channel[mask]
        else:
            # Старая логика для белого фона
            if color == 'red':
                colored[:,:,2] = gray_channel
            elif color == 'green':
                colored[:,:,1] = gray_channel
            elif color == 'blue':
                colored[:,:,0] = gray_channel
            elif color == 'yellow':
                colored[:,:,1] = gray_channel
                colored[:,:,2] = gray_channel
        
        return colored
    

    def draw_macrophages_on_composite(self, composite, macrophages_labels, color=(255, 255, 255), thickness=2):
        """Отрисовывает круги вокруг макрофагов на композите"""
        if composite is None or macrophages_labels is None:
            return composite
            
        try:
            composite_with_macrophages = composite.copy()
            
            # Находим центры макрофагов
            macrophages_props = measure.regionprops(macrophages_labels)
            
            for region in macrophages_props:
                # Получаем центр и радиус макрофага
                y, x = region.centroid
                radius = int(np.sqrt(region.area / np.pi)) + 5  # Увеличиваем радиус для макрофагов
                
                # Рисуем белый круг
                cv2.circle(composite_with_macrophages, (int(x), int(y)), radius, color, thickness)
                
            return composite_with_macrophages
            
        except Exception as e:
            print(f"   ❌ Ошибка отрисовки макрофагов: {e}")
            return composite


    def find_collagen_background_percentile(self, collagen_channel, percentile=90):
        """Находит яркость, при которой percentile% пикселей коллагена имеют меньшую яркость"""
        if collagen_channel is None:
            return 0
        
        try:
            # Вычисляем указанный перцентиль
            background_value = np.percentile(collagen_channel, percentile)
            
            # Диагностика
            total_pixels = collagen_channel.size
            pixels_below = np.sum(collagen_channel <= background_value)
            percentage_below = (pixels_below / total_pixels) * 100
            
            print(f"      📊 {percentile}% перцентиль коллагена: {background_value:.2f}")
            print(f"      📈 Пикселей ниже порога: {pixels_below}/{total_pixels} ({percentage_below:.1f}%)")
            
            return background_value
            
        except Exception as e:
            print(f"      ❌ Ошибка вычисления перцентиля: {e}")
            return np.percentile(collagen_channel, 50)  # fallback

    def subtract_vesicles_background_interactive(self, vesicles_channel, collagen_channel, sample_name):
        """Интерактивное вычитание фона везикул с рекомендацией и ручным вводом"""
        if vesicles_channel is None:
            return vesicles_channel
        
        try:
            # ⭐ НЕ ИЗМЕНЯЕМ КОЛЛАГЕН! Используем как есть
            collagen_percentile_value = np.percentile(collagen_channel, self.collagen_percentile) if collagen_channel is not None else 0
            
            # Анализируем распределение интенсивностей везикул
            vesicles_max = np.max(vesicles_channel)
            vesicles_mean = np.mean(vesicles_channel)
            
            print(f"\n   🔍 АНАЛИЗ ВЕЗИКУЛ ДЛЯ: {sample_name}")
            print(f"      📊 Коллаген (90%): {collagen_90th:.2f}")
            print(f"      📈 Везикулы - max: {vesicles_max:.2f}, mean: {vesicles_mean:.2f}")
            
            # Рекомендация
            if collagen_90th > 0 and vesicles_max > 0:
                recommended_percent = min(80, (collagen_90th / vesicles_max) * 100)
                recommended_percent = max(10, recommended_percent)
            else:
                recommended_percent = 30
            
            print(f"      💡 РЕКОМЕНДАЦИЯ: вычесть {recommended_percent:.1f}% от {collagen_90th:.2f}")
            print(f"      💡 Нажмите Enter для использования {recommended_percent:.1f}%")
            print(f"      💡 Или введите свой процент (0-100)")
            print(f"      💡 Или введите 'skip' для пропуска")
            
            # Интерактивный ввод
            while True:
                try:
                    user_input = "90".strip()
                    
                    if user_input.lower() == 'skip':
                        print("      ⏭️  Пропущено вычитание фона везикул")
                        return vesicles_channel
                    elif user_input == '':
                        subtraction_percent = recommended_percent / 100
                        break
                    else:
                        subtraction_percent = float(user_input) / 100
                        if 0 <= subtraction_percent <= 1:
                            break
                        else:
                            print("      ❌ Ошибка: введите значение от 0 до 100")
                except ValueError:
                    print("      ❌ Ошибка: введите числовое значение")
            
            # Применяем вычитание только если пользователь не пропустил
            amount_to_subtract = collagen_90th * subtraction_percent
            
            print(f"      🧮 Вычитание: {collagen_90th:.2f} × {subtraction_percent*100:.1f}% = {amount_to_subtract:.2f}")
            
            # Вычитаем и обрезаем отрицательные значения
            result = vesicles_channel.astype(np.float32) - amount_to_subtract
            result = np.clip(result, 0, 255).astype(np.uint8)
            
            after_max = np.max(result)
            after_mean = np.mean(result)
            print(f"      ✅ Везикулы после: max={after_max:.2f}, mean={after_mean:.2f}")
            
            return result
            
        except Exception as e:
            print(f"      ❌ Ошибка интерактивного вычитания: {e}")
            return vesicles_channel

    def create_composite_image(self, sample_data, output_dir):
        """Создание композитного изображения с правильным порядком каналов"""
        channels = sample_data['channels']
        sample_name = sample_data['sample_name']
        
        print(f"   🎨 Создание композита для: {sample_name}")
        print(f"   📊 Доступные каналы: {list(channels.keys())}")
        
        required_channels = [0, 2, 3]
        missing_channels = [ch for ch in required_channels if ch not in channels]
        if missing_channels:
            print(f"   ❌ Отсутствуют каналы: {missing_channels}")
            return None, None, None, None, None, None
        
        try:
            # Сохраняем СЫРЫЕ каналы
            nuclei_channel_raw = channels.get(0)      # Канал 0 - Ядра (СИНИЙ)
            collagen_channel_raw = channels.get(1)    # Канал 1 - Коллаген
            vesicles_channel_raw = channels.get(2)    # Канал 2 - Везикулы (КРАСНЫЙ)
            protein_channel_raw = channels.get(3)     # Канал 3 - Белок (ЗЕЛЕНЫЙ)
            
            # Сохраняем оригинальный канал везикул
            vesicles_channel_original = vesicles_channel_raw.copy()
            
            # ========== ШАГ 1: СЕГМЕНТАЦИЯ ВЕЗИКУЛ ==========
            print("   🔍 Сегментация везикул на оригинальном канале...")
            vesicles_binary, vesicle_stats = self.segment_vesicles_with_criteria(
                vesicles_channel_original, collagen_channel_raw, sample_name
            )
            
            # ========== ШАГ 2: ПОДГОТОВКА КАНАЛОВ ==========
            print("   🎨 Подготовка каналов для визуализации...")
            
            # --- ВЕЗИКУЛЫ (КРАСНЫЙ) ---
            vesicles_channel_visual = vesicles_channel_original.copy()
            
            # Вычитание коллагена с защитой везикул
            if collagen_channel_raw is not None:
                print(f"      🧹 Вычитание коллагена из фона...")
                
                # Создаем защитную маску
                if vesicles_binary is not None and np.sum(vesicles_binary) > 0:
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    protected_mask = cv2.dilate(vesicles_binary.astype(np.uint8), kernel, iterations=1).astype(bool)
                    bright_pixels = vesicles_channel_original > self.min_vesicle_intensity * 0.8
                    protected_mask = protected_mask | bright_pixels
                else:
                    protected_mask = vesicles_channel_original > self.min_vesicle_intensity * 0.8
                
                background_mask = ~protected_mask
                
                # Вычитание
                vesicles_float = vesicles_channel_visual.astype(np.float32)
                collagen_float = collagen_channel_raw.astype(np.float32)
                subtraction_amount = collagen_float * self.subtraction_factor
                vesicles_float[background_mask] = np.maximum(0, vesicles_float[background_mask] - subtraction_amount[background_mask])
                vesicles_channel_visual = np.clip(vesicles_float, 0, 255).astype(np.uint8)
            
            # Предобработка везикул
            vesicles_channel_visual = self.preprocess_channel(vesicles_channel_visual, 2)
            vesicles_channel_visual = self.enhance_vesicles_brightness(vesicles_channel_visual)
            vesicles_channel_visual = self.enhance_contrast(vesicles_channel_visual, max(1.0, self.vesicles_contrast_factor))
            
            # --- БЕЛОК (ЗЕЛЕНЫЙ) ---
            print("   🔧 Обработка белка по эталону...")

            # Сначала предобработка сырого канала белка
            protein_channel = self.preprocess_channel(protein_channel_raw, 3)

            # Сохраняем ДО IgG коррекции для сравнения
            protein_channel_before_correction = protein_channel.copy()

            # Получаем IgG фон
            protein_igg_background = self.get_protein_igg_background()

            if protein_igg_background > 0:
                print("      • Применяем IgG коррекцию...")
                
                protein_before_max = np.max(protein_channel)
                protein_before_mean = np.mean(protein_channel)
                
                print(f"      • IgG фон: {protein_igg_background:.2f}")
                print(f"      • Белок до коррекции: max={protein_before_max:.2f}, mean={protein_before_mean:.2f}")
                
                # Вычитаем процент от IgG фона (из параметров)
                subtraction_fraction = self.protein_subtraction_percent / 100.0
                amount_to_subtract = protein_igg_background * subtraction_fraction
                
                # Защита от слишком агрессивного вычитания
                if amount_to_subtract >= protein_before_max * 0.8:
                    amount_to_subtract = protein_before_max * 0.5
                    print(f"      ⚠️ СЛИШКОМ МНОГО! Снижено до: {amount_to_subtract:.2f}")
                
                # Вычитаем из предобработанного канала
                protein_channel_corrected = protein_channel.astype(np.float32) - amount_to_subtract
                protein_channel_corrected = np.clip(protein_channel_corrected, 0, 255).astype(np.uint8)
                
                protein_after_max = np.max(protein_channel_corrected)
                protein_after_mean = np.mean(protein_channel_corrected)
                
                print(f"      • Вычтено: {amount_to_subtract:.2f} ({self.protein_subtraction_percent}% от IgG фона)")
                print(f"      • Белок после коррекции: max={protein_after_max:.2f}, mean={protein_after_mean:.2f}")
                
                # Применяем яркость к скорректированному каналу
                if self.protein_brightness_factor != 1.0:
                    print(f"      • Коэффициент яркости: {self.protein_brightness_factor}")
                    protein_float = protein_channel_corrected.astype(np.float32)
                    protein_brightened = protein_float * self.protein_brightness_factor
                    protein_brightened = np.clip(protein_brightened, 0, 255)
                    protein_channel_visual = protein_brightened.astype(np.uint8)
                    
                    before_max = np.max(protein_float)
                    after_max = np.max(protein_brightened)
                    print(f"      • Яркость белка: {before_max:.1f} → {after_max:.1f}")
                else:
                    protein_channel_visual = protein_channel_corrected
                    print(f"      • Яркость белка не изменена")
            else:
                print(f"      • IgG фон не найден, используем предобработанный канал")
                protein_channel_visual = protein_channel

            # Применяем контраст (как в эталоне)
            protein_channel_visual = self.enhance_contrast(protein_channel_visual, max(1.0, self.protein_contrast_factor))
            
            # --- ЯДРА (СИНИЙ) ---
            nuclei_channel_visual = self.preprocess_channel(nuclei_channel_raw, 0)
            
            # ========== СОЗДАНИЕ RGB КОМПОЗИТА ==========
            print("   🖼️ Создание RGB композита с правильными цветами и наложением:")
            print("      • Зеленый (задний план)  - Белок")
            print("      • Синий   (средний план) - Ядра")
            print("      • Красный (передний план) - Везикулы")

            height, width = protein_channel_visual.shape
            composite = np.zeros((height, width, 3), dtype=np.uint8)

            # ШАГ 1: Сначала добавляем белок (зеленый) на задний план
            composite[:,:,1] = protein_channel_visual  # Зеленый канал - Белок

            # ШАГ 2: Поверх добавляем ядра (синие) - они будут перекрывать белок где есть сигнал
            # Но нужно добавить их ТОЛЬКО там, где есть ядра, иначе они сделают фон синим
            nuclei_mask = nuclei_channel_visual > 20  # Порог для ядер
            composite[nuclei_mask, 0] = nuclei_channel_visual[nuclei_mask]  # Синий канал - Ядра (только где есть сигнал)

            # ШАГ 3: Сверху добавляем везикулы (красные)
            vesicles_mask = vesicles_channel_visual > 10  # Порог для везикул
            composite[vesicles_mask, 2] = vesicles_channel_visual[vesicles_mask]  # Красный канал - Везикулы (только где есть сигнал)

            print(f"   📊 Статистика наложения:")
            print(f"      • Пикселей с белком: {np.sum(composite[:,:,1] > 0)}")
            print(f"      • Пикселей с ядрами: {np.sum(composite[:,:,0] > 0)}")
            print(f"      • Пикселей с везикулами: {np.sum(composite[:,:,2] > 0)}")
            
            # ========== СОХРАНЕНИЕ ==========
            safe_name = "".join(c for c in sample_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            
            # Сохраняем оригинальные каналы
            self.save_original_channels(
                vesicles_channel_original, 
                protein_channel_raw, 
                nuclei_channel_raw,
                collagen_channel_raw,
                safe_name, 
                output_dir
            )
            
            # Сохраняем обработанные каналы
            self.save_processed_channels(
                vesicles_channel_visual, 
                protein_channel_visual,
                nuclei_channel_visual, 
                composite, 
                safe_name, 
                output_dir
            )
            
            # Сегментируем макрофаги для анализа
            print(f"   🔍 Сегментация макрофагов...")
            macrophages_labels = self.segment_macrophages(nuclei_channel_visual, protein_channel_visual)
            
            # Сохраняем аннотированные изображения
            if vesicles_binary is not None and np.sum(vesicles_binary) > 0:
                self.save_annotated_images(
                    composite, 
                    vesicles_binary, 
                    macrophages_labels,
                    vesicles_channel_visual, 
                    safe_name, 
                    output_dir
                )
            
            # ⭐ ДИАГНОСТИКА КОЛОКАЛИЗАЦИИ
            if vesicles_binary is not None and macrophages_labels is not None:
                vesicle_pixels = np.sum(vesicles_binary > 0)
                macrophage_pixels = np.sum(macrophages_labels > 0)
                intersection = np.sum((vesicles_binary > 0) & (macrophages_labels > 0))
                
                print(f"   📊 ДИАГНОСТИКА КОЛОКАЛИЗАЦИИ:")
                print(f"      • Пикселей везикул: {vesicle_pixels}")
                print(f"      • Пикселей макрофагов: {macrophage_pixels}")
                print(f"      • Пересечение: {intersection}")
                if vesicle_pixels > 0:
                    print(f"      • % везикул в макрофагах: {intersection/vesicle_pixels*100:.2f}%")
            
            print(f"   ✅ Все изображения сохранены")
            
            return composite, nuclei_channel_visual, vesicles_binary, protein_channel_visual, f"{safe_name}_composite.png", vesicles_channel_original
            
        except Exception as e:
            print(f"   ❌ Ошибка создания композита: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None, None, None, None
    
    def segment_vesicles(self, vesicles_channel):
        """Улучшенная сегментация везикул - ВСЯ красная площадь делится на везикулы"""
        if vesicles_channel is None:
            return None
            
        try:
            print("   🔍 УЛУЧШЕННАЯ СЕГМЕНТАЦИЯ ВЕЗИКУЛ...")
            
            # 1. Получаем ВСЮ красную площадь (после коррекции)
            # Используем адаптивный порог чтобы захватить всю область везикул
            try:
                # Метод 1: Otsu для автоматического определения порога
                otsu_threshold = filters.threshold_otsu(vesicles_channel)
                binary_otsu = vesicles_channel > otsu_threshold
            except:
                # Метод 2: Фиксированный порог если Otsu не работает
                binary_otsu = vesicles_channel > 10
            intensity_mask = vesicles_channel > self.min_vesicle_intensity

            
            # Метод 3: Перцентиль чтобы захватить больше области
            percentile_threshold = np.percentile(vesicles_channel[vesicles_channel > 0], 30) if np.any(vesicles_channel > 0) else 10
            binary_percentile = vesicles_channel > percentile_threshold
            
            # Объединяем маски чтобы получить ВСЮ красную площадь
            full_red_area = np.logical_and(binary_otsu, intensity_mask)
            
            # Морфологическое закрытие чтобы заполнить мелкие дыры
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            full_red_area = cv2.morphologyEx(full_red_area.astype(np.uint8), cv2.MORPH_CLOSE, kernel_close)
            full_red_area = full_red_area.astype(bool)
            
            # Заполняем отверстия
            full_red_area = ndimage.binary_fill_holes(full_red_area)
            
            total_red_pixels = np.sum(full_red_area)
            print(f"      📏 Вся красная площадь: {total_red_pixels} пикселей")
            
            if total_red_pixels == 0:
                print("      ⚠️  Нет красной площади для сегментации")
                return np.zeros_like(vesicles_channel, dtype=np.uint8)
            
            # 2. Сегментируем ВСЮ красную площадь на везикулы
            final_vesicles = self.segment_entire_red_area(full_red_area, vesicles_channel)
            
            # 3. Подсчет результатов
            final_labels = measure.label(final_vesicles)
            final_count = np.max(final_labels) if np.max(final_labels) > 0 else 0
            
            # Анализ размеров
            regions = measure.regionprops(final_labels)
            if regions:
                areas = [r.area for r in regions]
                avg_area = np.mean(areas)
                min_area = np.min(areas)
                max_area = np.max(areas)
                print(f"   📊 РЕЗУЛЬТАТ СЕГМЕНТАЦИИ:")
                print(f"      • Всего везикул: {final_count}")
                print(f"      • Размеры: min={min_area:.1f}, avg={avg_area:.1f}, max={max_area:.1f}")
                print(f"      • Покрытие: {total_red_pixels}px → {sum(areas)}px ({sum(areas)/total_red_pixels*100:.1f}%)")
            else:
                print(f"   ⚠️  Не создано везикул")
            
            return final_vesicles
                
        except Exception as e:
            print(f"   ❌ Ошибка сегментации везикул: {e}")
            import traceback
            traceback.print_exc()
            return None

    def segment_entire_red_area(self, full_red_area, vesicles_channel):
        """Сегментирует ВСЮ красную площадь на везикулы оптимального размера"""
        try:
            print("      🎯 Сегментация всей красной площади...")
            
            # Проверка входных данных
            if full_red_area is None or vesicles_channel is None:
                print("      ⚠️  Пустые входные данные")
                return np.zeros_like(full_red_area, dtype=np.uint8)
            
            if not np.any(full_red_area):
                print("      ⚠️  Нет красной площади для сегментации")
                return np.zeros_like(full_red_area, dtype=np.uint8)
            
            # Метка всей красной области
            red_labels = measure.label(full_red_area)
            red_regions = measure.regionprops(red_labels, intensity_image=vesicles_channel)
            
            # Целевой размер везикулы (средний между min и max)
            target_vesicle_size = (self.min_vesicle_size + self.max_vesicle_size) // 2
            print(f"      🎯 Целевой размер везикулы: {target_vesicle_size}px")
            
            # Итоговая маска всех везикул
            final_vesicles_mask = np.zeros_like(full_red_area, dtype=bool)
            
            total_clusters = 0
            total_created_vesicles = 0
            
            for i, region in enumerate(red_regions):
                try:
                    region_area = region.area
                    region_mask = red_labels == (i + 1)
                    
                    print(f"      🔍 Область {i+1}: {region_area}px")
                    
                    if region_area < self.min_vesicle_size:
                        # Слишком маленькая - пропускаем
                        continue
                        
                    elif region_area <= self.max_vesicle_size:
                        # ⭐ ИСПРАВЛЕНИЕ: Проверяем форму области!
                        # Даже если по площади проходит, может быть неправильной формы
                        # Используем "круглость" как критерий
                        y0, x0 = region.bbox[0], region.bbox[1]
                        y1, x1 = region.bbox[2], region.bbox[3]
                        bbox_width = x1 - x0
                        bbox_height = y1 - y0
                        
                        # Вычисляем эквивалентный диаметр (диаметр круга такой же площади)
                        equivalent_diameter = 2 * np.sqrt(region_area / np.pi)
                        
                        # Отношение сторон bounding box
                        aspect_ratio = max(bbox_width, bbox_height) / min(bbox_width, bbox_height) if min(bbox_width, bbox_height) > 0 else 1
                        
                        # Если область достаточно "круглая" и аспектное отношение не слишком большое
                        if aspect_ratio < 2.0 and equivalent_diameter < 2 * np.sqrt(self.max_vesicle_size / np.pi):
                            # Оставляем как одну везикулу
                            final_vesicles_mask[region_mask] = True
                            total_created_vesicles += 1
                            print(f"         ✅ Круглая везикула: {region_area}px (aspect={aspect_ratio:.1f})")
                        else:
                            # ⭐ НОВОЕ: Даже если по площади проходит, но форма неправильная - разделяем!
                            total_clusters += 1
                            print(f"         🔄 Неправильная форма: {region_area}px (aspect={aspect_ratio:.1f}) → сегментируем...")
                            
                            # Разделяем область на оптимальные везикулы
                            cluster_vesicles = self.optimal_cluster_segmentation(
                                region_mask, region_area, target_vesicle_size, vesicles_channel
                            )
                            
                            cluster_count = np.max(measure.label(cluster_vesicles)) if np.any(cluster_vesicles) else 0
                            final_vesicles_mask[cluster_vesicles] = True
                            total_created_vesicles += cluster_count
                            
                            print(f"         ✅ Создано {cluster_count} везикул из области неправильной формы")
                            
                    else:
                        # Большая область - ИСКУСТВЕННО разделяем на везикулы
                        total_clusters += 1
                        print(f"         🎯 Большое скопление: {region_area}px → сегментируем...")
                        
                        # Разделяем область на оптимальные везикулы
                        cluster_vesicles = self.optimal_cluster_segmentation(
                            region_mask, region_area, target_vesicle_size, vesicles_channel
                        )
                        
                        cluster_count = np.max(measure.label(cluster_vesicles)) if np.any(cluster_vesicles) else 0
                        final_vesicles_mask[cluster_vesicles] = True
                        total_created_vesicles += cluster_count
                        
                        print(f"         ✅ Создано {cluster_count} везикул из скопления")
                        
                except Exception as region_error:
                    print(f"         ❌ Ошибка обработки области {i+1}: {region_error}")
                    continue  # Продолжаем обработку других областей
            
            # Пост-обработка: убираем слишком мелкие объекты
            try:
                final_cleaned = morphology.remove_small_objects(
                    final_vesicles_mask, 
                    min_size=self.min_vesicle_size
                )
                
                # Морфологическое открытие для разделения слабо соединенных везикул
                kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
                final_cleaned = cv2.morphologyEx(final_cleaned.astype(np.uint8), cv2.MORPH_OPEN, kernel_open)
                
                final_pixels = np.sum(final_cleaned)
                original_pixels = np.sum(full_red_area)
                coverage = final_pixels / original_pixels * 100 if original_pixels > 0 else 0
                
                print(f"      📊 ИТОГ СЕГМЕНТАЦИИ:")
                print(f"         • Обработано областей: {len(red_regions)}")
                print(f"         • Больших скоплений: {total_clusters}")
                print(f"         • Создано везикул: {total_created_vesicles}")
                print(f"         • Покрытие: {final_pixels}/{original_pixels}px ({coverage:.1f}%)")
                
                return final_cleaned.astype(np.uint8) * 255
                
            except Exception as postprocess_error:
                print(f"      ❌ Ошибка пост-обработки: {postprocess_error}")
                # Возвращаем то, что получилось, без очистки
                return final_vesicles_mask.astype(np.uint8) * 255
                
        except Exception as main_error:
            print(f"      ❌ Критическая ошибка сегментации красной площади: {main_error}")
            import traceback
            traceback.print_exc()
            
            # Возвращаем пустую маску в случае критической ошибки
            if full_red_area is not None:
                return np.zeros_like(full_red_area, dtype=np.uint8)
            else:
                # Если даже full_red_area is None, возвращаем пустой массив
                return np.array([], dtype=np.uint8)

    def optimal_cluster_segmentation(self, cluster_mask, cluster_area, target_size, vesicles_channel):
        """
        Оптимальное разделение скопления на везикулы с учетом интенсивности
        
        Args:
            cluster_mask: бинарная маска скопления
            cluster_area: площадь скопления в пикселях
            target_size: целевой размер везикулы
            vesicles_channel: исходный канал с интенсивностями
        
        Returns:
            binary_mask: бинарная маска разделенных везикул
        """
        try:
            # Вычисляем оптимальное количество везикул
            optimal_count = max(2, int(cluster_area / target_size))
            
            # Находим bounding box скопления
            coords = np.where(cluster_mask)
            if len(coords[0]) == 0 or len(coords[1]) == 0:
                print(f"            ⚠️  Пустое скопление")
                return np.zeros_like(cluster_mask, dtype=bool)
            
            y_min, y_max = np.min(coords[0]), np.max(coords[0])
            x_min, x_max = np.min(coords[1]), np.max(coords[1])
            height, width = y_max - y_min + 1, x_max - x_min + 1
            
            print(f"            📐 BBox: {width}x{height}, площадь: {cluster_area}px")
            print(f"            🎯 Целевое количество везикул: {optimal_count} (по {target_size}px каждая)")
            print(f"            🎯 Мин. интенсивность: {self.min_vesicle_intensity}")
            
            # ⭐ МЕТОД 1: Watershed на основе интенсивности
            result_mask = self.watershed_intensity_segmentation(
                cluster_mask, vesicles_channel, optimal_count
            )
            
            # Проверяем результат метода 1
            if result_mask is not None and np.sum(result_mask) > 0:
                # Проверяем среднюю интенсивность найденных везикул
                vesicle_intensities = vesicles_channel[result_mask]
                if len(vesicle_intensities) > 0:
                    mean_intensity = np.mean(vesicle_intensities)
                    print(f"            📊 Метод 1: средняя интенсивность = {mean_intensity:.1f}")
                    
                    if mean_intensity < self.min_vesicle_intensity:
                        print(f"            ⚠️  Слишком тусклые везикулы (ниже порога {self.min_vesicle_intensity})")
                        result_mask = np.zeros_like(cluster_mask, dtype=bool)
            
            result_area = np.sum(result_mask)
            coverage = result_area / cluster_area if cluster_area > 0 else 0
            
            # Если метод 1 дал плохой результат, пробуем метод 2
            if coverage < 0.6 or result_area == 0:  # Меньше 60% покрытия
                print(f"            🔄 Метод 1 дал плохое покрытие ({coverage*100:.1f}%), пробуем метод 2...")
                
                # ⭐ МЕТОД 2: Адаптивная сетка на основе интенсивности
                result_mask = self.adaptive_grid_segmentation(
                    cluster_mask, vesicles_channel, y_min, y_max, x_min, x_max, 
                    optimal_count, target_size
                )
                
                # Проверяем результат метода 2
                if result_mask is not None and np.sum(result_mask) > 0:
                    vesicle_intensities = vesicles_channel[result_mask]
                    mean_intensity = np.mean(vesicle_intensities)
                    print(f"            📊 Метод 2: средняя интенсивность = {mean_intensity:.1f}")
                    
                    if mean_intensity < self.min_vesicle_intensity:
                        print(f"            ⚠️  Слишком тусклые везикулы (ниже порога {self.min_vesicle_intensity})")
                        result_mask = np.zeros_like(cluster_mask, dtype=bool)
                
                result_area = np.sum(result_mask)
                coverage = result_area / cluster_area if cluster_area > 0 else 0
                
                # Если метод 2 тоже плох, пробуем метод 3
                if coverage < 0.6 or result_area == 0:
                    print(f"            🔄 Метод 2 тоже плох ({coverage*100:.1f}%), используем равномерную сетку")
                    
                    # ⭐ МЕТОД 3: Равномерная сетка (гарантированный результат)
                    result_mask = self.uniform_grid_segmentation_improved(
                        cluster_mask, cluster_area, target_size, y_min, y_max, x_min, x_max
                    )
                    
                    # Финальная проверка интенсивности
                    if result_mask is not None and np.sum(result_mask) > 0:
                        vesicle_intensities = vesicles_channel[result_mask]
                        mean_intensity = np.mean(vesicle_intensities)
                        print(f"            📊 Метод 3: средняя интенсивность = {mean_intensity:.1f}")
            
            # Финальная обработка: удаляем слишком тусклые везикулы
            if result_mask is not None and np.sum(result_mask) > 0:
                # Маркируем отдельные везикулы
                labels = measure.label(result_mask)
                regions = measure.regionprops(labels, intensity_image=vesicles_channel)
                
                # Создаем новую маску только для достаточно ярких везикул
                final_mask = np.zeros_like(result_mask, dtype=bool)
                bright_vesicles = 0
                dim_vesicles = 0
                
                for region in regions:
                    mean_intensity = region.mean_intensity
                    
                    if mean_intensity >= self.min_vesicle_intensity:
                        final_mask[labels == region.label] = True
                        bright_vesicles += 1
                    else:
                        dim_vesicles += 1
                        print(f"            ⚠️  Отброшена тусклая везикула: средняя интенсивность {mean_intensity:.1f} < {self.min_vesicle_intensity}")
                
                result_mask = final_mask
                
                print(f"            📊 ИТОГ:")
                print(f"               • Ярких везикул: {bright_vesicles}")
                print(f"               • Отброшено тусклых: {dim_vesicles}")
            
            # Финальная очистка
            if result_mask is not None and np.sum(result_mask) > 0:
                final_vesicles = self.finalize_vesicles(result_mask, target_size)
                
                final_coverage = np.sum(final_vesicles) / cluster_area * 100 if cluster_area > 0 else 0
                final_count = np.max(measure.label(final_vesicles)) if np.any(final_vesicles) else 0
                
                print(f"            ✅ Результат: {final_count} везикул, покрытие: {final_coverage:.1f}%")
                
                return final_vesicles
            else:
                print(f"            ⚠️  Не удалось создать везикулы")
                return np.zeros_like(cluster_mask, dtype=bool)
            
        except Exception as e:
            print(f"            ❌ Ошибка оптимальной сегментации: {e}")
            import traceback
            traceback.print_exc()
            # Резервный метод: равномерная сетка
            return self.uniform_grid_segmentation(cluster_mask, cluster_area, target_size)

    def adaptive_grid_segmentation(self, cluster_mask, vesicles_channel, y_min, y_max, x_min, x_max, optimal_count, target_size):
        """Адаптивное разделение на основе интенсивности"""
        try:
            height, width = y_max - y_min + 1, x_max - x_min + 1
            
            # Анализируем распределение интенсивности внутри скопления
            # ⭐ ИСПРАВЛЕНИЕ: создаем маску для области кластера
            cluster_region = np.zeros_like(vesicles_channel, dtype=bool)
            cluster_region[y_min:y_max+1, x_min:x_max+1] = cluster_mask[y_min:y_max+1, x_min:x_max+1]
            
            # Получаем интенсивности только внутри кластера
            cluster_intensities = vesicles_channel[cluster_region]
            
            if len(cluster_intensities) == 0:
                return self.uniform_grid_segmentation(cluster_mask, np.sum(cluster_mask), target_size)
            
            # Находим области с высокой интенсивностью (центры везикул)
            high_intensity_threshold = np.percentile(cluster_intensities, 70)
            high_intensity_mask = np.zeros_like(vesicles_channel, dtype=bool)
            
            # ⭐ ИСПРАВЛЕНИЕ: правильно создаем маску высокой интенсивности
            # Только пиксели внутри кластера И выше порога
            high_intensity_pixels = (vesicles_channel[y_min:y_max+1, x_min:x_max+1] > high_intensity_threshold) & cluster_mask[y_min:y_max+1, x_min:x_max+1]
            high_intensity_mask[y_min:y_max+1, x_min:x_max+1] = high_intensity_pixels
            
            # Если нашли яркие области, используем их как центры
            if np.sum(high_intensity_mask) > 0:
                # Находим связные компоненты ярких областей
                bright_labels = measure.label(high_intensity_mask)
                bright_regions = measure.regionprops(bright_labels)
                
                if len(bright_regions) >= optimal_count * 0.5:  # Достаточно ярких центров
                    print("            💡 Используем яркие центры для разделения")
                    return self.segment_around_bright_centers(
                        cluster_mask, bright_regions, target_size
                    )
            
            # Если ярких центров мало, используем интеллектуальную сетку
            print("            💡 Используем интеллектуальную сетку")
            return self.smart_grid_segmentation(
                cluster_mask, vesicles_channel, y_min, y_max, x_min, x_max, optimal_count, target_size
            )
            
        except Exception as e:
            print(f"            ❌ Ошибка адаптивной сетки: {e}")
            import traceback
            traceback.print_exc()
            return self.uniform_grid_segmentation(cluster_mask, np.sum(cluster_mask), target_size)

    def smart_grid_segmentation(self, cluster_mask, vesicles_channel, y_min, y_max, x_min, x_max, optimal_count, target_size):
        """Интеллектуальная сетка с учетом интенсивности"""
        try:
            height, width = y_max - y_min + 1, x_max - x_min + 1
            
            # Определяем форму сетки на основе пропорций области
            aspect_ratio = width / height if height > 0 else 1
            cols = max(2, int(np.sqrt(optimal_count * aspect_ratio)))
            rows = max(2, int(optimal_count / cols))
            
            # Корректируем чтобы общее количество было близко к оптимальному
            while cols * rows < optimal_count * 0.8:
                if width > height:
                    cols += 1
                else:
                    rows += 1
            
            print(f"            📐 Интеллектуальная сетка: {rows}x{cols}")
            
            result_mask = np.zeros_like(cluster_mask, dtype=bool)
            cell_height = max(target_size // 2, height // rows)
            cell_width = max(target_size // 2, width // cols)
            
            print(f"            📏 Размер ячейки: {cell_height}x{cell_width}")
            
            for i in range(rows):
                for j in range(cols):
                    y_start = y_min + i * cell_height
                    y_end = min(y_max, y_start + cell_height)
                    x_start = x_min + j * cell_width  
                    x_end = min(x_max, x_start + cell_width)
                    
                    # Проверяем что индексы корректны
                    if y_end <= y_start or x_end <= x_start:
                        continue
                    
                    # Проверяем, что ячейка вообще попадает в кластер
                    cell_slice_y = slice(y_start, y_end)
                    cell_slice_x = slice(x_start, x_end)
                    
                    # Создаем маску для текущей ячейки из полного кластера
                    cell_cluster_mask = cluster_mask[cell_slice_y, cell_slice_x]
                    
                    # Если в ячейке нет пикселей кластера, пропускаем
                    if not np.any(cell_cluster_mask):
                        continue
                        
                    # Пропорциональный эллипс вместо прямоугольника
                    center_y = (y_start + y_end) // 2
                    center_x = (x_start + x_end) // 2
                    radius_y = max(3, cell_height // 2 - 1)  # Уменьшаем радиус чтобы избежать перекрытия
                    radius_x = max(3, cell_width // 2 - 1)
                    
                    # Создаем эллиптическую маску в локальных координатах ячейки
                    cell_height_local = y_end - y_start
                    cell_width_local = x_end - x_start
                    yy_local, xx_local = np.ogrid[0:cell_height_local, 0:cell_width_local]
                    
                    # Центр эллипса в локальных координатах
                    center_y_local = cell_height_local // 2
                    center_x_local = cell_width_local // 2
                    
                    ellipse_mask_local = ((xx_local - center_x_local) ** 2 / (radius_x ** 2) + 
                                        (yy_local - center_y_local) ** 2 / (radius_y ** 2)) <= 1
                    
                    # Пересекаем эллипс с кластером в пределах ячейки
                    final_cell_mask_local = ellipse_mask_local & cell_cluster_mask
                    
                    # Если есть пересечение, добавляем в результат
                    if np.any(final_cell_mask_local):
                        # Создаем маску в глобальных координатах
                        cell_region = np.zeros_like(cluster_mask, dtype=bool)
                        cell_region[cell_slice_y, cell_slice_x] = final_cell_mask_local
                        
                        # Проверяем размер
                        cell_area = np.sum(final_cell_mask_local)
                        if self.min_vesicle_size <= cell_area <= self.max_vesicle_size:
                            result_mask[cell_region] = True
                            print(f"               ✅ Ячейка ({i},{j}): площадь={cell_area}px")
            
            total_area = np.sum(result_mask)
            if total_area > 0:
                print(f"            📊 Создано везикул: {total_area/target_size:.1f} (покрытие: {total_area}px)")
            
            return result_mask
            
        except Exception as e:
            print(f"            ❌ Ошибка интеллектуальной сетки: {e}")
            import traceback
            traceback.print_exc()
            return self.uniform_grid_segmentation(cluster_mask, np.sum(cluster_mask), target_size)

    def segment_around_bright_centers(self, cluster_mask, bright_regions, target_size):
        """Разделение вокруг ярких центров"""
        result_mask = np.zeros_like(cluster_mask, dtype=bool)
        
        for region in bright_regions:
            center_y, center_x = map(int, region.centroid)
            
            # ⭐ ПРОВЕРКА: что центр находится внутри кластера
            if not (0 <= center_y < cluster_mask.shape[0] and 0 <= center_x < cluster_mask.shape[1]):
                continue
                
            if not cluster_mask[center_y, center_x]:
                continue
            
            # Создаем круглую везикулу вокруг центра
            max_radius = int(np.sqrt(self.max_vesicle_size / np.pi))
            radius = min(int(np.sqrt(target_size / np.pi)), max_radius)
            
            # Создаем круглую маску
            y_start = max(0, center_y - radius)
            y_end = min(cluster_mask.shape[0], center_y + radius + 1)
            x_start = max(0, center_x - radius)
            x_end = min(cluster_mask.shape[1], center_x + radius + 1)
            
            # ⭐ ИСПРАВЛЕНИЕ: создаем правильные индексы для круга
            yy_local, xx_local = np.ogrid[y_start:y_end, x_start:x_end]
            circle_mask = (xx_local - center_x) ** 2 + (yy_local - center_y) ** 2 <= radius ** 2
            
            # Пересекаем с исходным скоплением
            cell_region = np.zeros_like(cluster_mask, dtype=bool)
            cell_region[y_start:y_end, x_start:x_end] = circle_mask
            
            final_mask = cell_region & cluster_mask
            
            if np.sum(final_mask) >= self.min_vesicle_size:
                result_mask[final_mask] = True
        
        return result_mask

    def uniform_grid_segmentation(self, cluster_mask, cluster_area, target_size):
        """Равномерное разделение как резервный метод"""
        optimal_count = max(2, int(cluster_area / target_size))
        
        # Находим bounding box
        coords = np.where(cluster_mask)
        y_min, y_max = np.min(coords[0]), np.max(coords[0])
        x_min, x_max = np.min(coords[1]), np.max(coords[1])
        height, width = y_max - y_min + 1, x_max - x_min + 1
        
        # Простая квадратная сетка
        grid_size = int(np.ceil(np.sqrt(optimal_count)))
        cell_size = max(self.min_vesicle_size // 2, min(height, width) // grid_size)
        
        result_mask = np.zeros_like(cluster_mask, dtype=bool)
        
        for i in range(grid_size):
            for j in range(grid_size):
                y_center = y_min + (i + 0.5) * cell_size
                x_center = x_min + (j + 0.5) * cell_size
                
                if y_center >= cluster_mask.shape[0] or x_center >= cluster_mask.shape[1]:
                    continue
                    
                radius = cell_size // 2 - 1
                y_start, y_end = int(max(0, y_center - radius)), int(min(cluster_mask.shape[0], y_center + radius))
                x_start, x_end = int(max(0, x_center - radius)), int(min(cluster_mask.shape[1], x_center + radius))
                
                if y_end <= y_start or x_end <= x_start:
                    continue
                
                # Круглая маска в локальных координатах
                cell_height = y_end - y_start
                cell_width = x_end - x_start
                yy_local, xx_local = np.ogrid[0:cell_height, 0:cell_width]
                
                # Центр в локальных координатах
                center_y_local = cell_height // 2
                center_x_local = cell_width // 2
                
                circle_mask_local = (xx_local - center_x_local) ** 2 + (yy_local - center_y_local) ** 2 <= radius ** 2
                
                # Берем срез кластера
                cell_cluster_mask = cluster_mask[y_start:y_end, x_start:x_end]
                
                # Пересекаем
                final_mask_local = circle_mask_local & cell_cluster_mask
                
                if np.any(final_mask_local):
                    cell_region = np.zeros_like(cluster_mask, dtype=bool)
                    cell_region[y_start:y_end, x_start:x_end] = final_mask_local
                    
                    if np.sum(final_mask_local) >= self.min_vesicle_size:
                        result_mask[cell_region] = True
        
        return result_mask
    
    def segment_macrophages(self, nuclei_channel, protein_channel):
        """Сегментация макрофагов"""
        if nuclei_channel is None or protein_channel is None:
            return None
            
        try:
            print("   🔍 Сегментация макрофагов...")
            
            # Диагностика интенсивностей
            print(f"      • Ядра - min={np.min(nuclei_channel):.1f}, max={np.max(nuclei_channel):.1f}, mean={np.mean(nuclei_channel):.1f}")
            print(f"      • Белок - min={np.min(protein_channel):.1f}, max={np.max(protein_channel):.1f}, mean={np.mean(protein_channel):.1f}")
            
            nuclei_threshold = filters.threshold_otsu(nuclei_channel)
            nuclei_binary = nuclei_channel > nuclei_threshold
            
            protein_threshold = filters.threshold_otsu(protein_channel)
            protein_binary = protein_channel > protein_threshold
            
            print(f"      • Порог ядер (Otsu): {nuclei_threshold:.1f}")
            print(f"      • Порог белка (Otsu): {protein_threshold:.1f}")
            print(f"      • Пикселей ядер: {np.sum(nuclei_binary)} ({np.sum(nuclei_binary)/nuclei_binary.size*100:.1f}%)")
            print(f"      • Пикселей белка: {np.sum(protein_binary)} ({np.sum(protein_binary)/protein_binary.size*100:.1f}%)")
            
            macrophages_binary = np.logical_and(nuclei_binary, protein_binary)
            print(f"      • Пересечение (кандидаты в макрофаги): {np.sum(macrophages_binary)} пикселей")
            
            if np.sum(macrophages_binary) == 0:
                print(f"      ⚠️ Нет пересечения ядер и белка!")
                return None
            
            macrophages_binary = ndimage.binary_fill_holes(macrophages_binary)
            macrophages_binary = morphology.remove_small_objects(macrophages_binary, min_size=50)
            
            distance = ndimage.distance_transform_edt(macrophages_binary)
            coordinates = peak_local_max(distance, min_distance=10, labels=macrophages_binary)
            
            if len(coordinates) == 0:
                print(f"      ⚠️ Не найдено локальных максимумов для watershed")
                # Fallback: просто connected components
                labels = measure.label(macrophages_binary)
            else:
                local_maxi = np.zeros_like(distance, dtype=bool)
                local_maxi[tuple(coordinates.T)] = True
                markers = measure.label(local_maxi)
                labels = segmentation.watershed(-distance, markers, mask=macrophages_binary)
            
            macrophage_count = np.max(labels) if np.max(labels) > 0 else 0
            print(f"   ✅ Сегментировано макрофагов: {macrophage_count}")
            
            return labels
            
        except Exception as e:
            print(f"   ❌ Ошибка сегментации макрофагов: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def analyze_colocalization(self, vesicles_binary, macrophages_labels, nuclei_channel, sample_name):
        """Анализ колокализации везикул и макрофагов + везикул в клетках"""
        if vesicles_binary is None or macrophages_labels is None:
            return 0, 0, 0, 0, []  # ⭐ ВОЗВРАЩАЕМ 5 ЗНАЧЕНИЙ
        
        try:
            vesicles_labels = measure.label(vesicles_binary)
            vesicles_props = measure.regionprops(vesicles_labels)
            
            total_vesicles = len(vesicles_props)
            colocalized_vesicles = 0
            colocalization_data = []
            
            print(f"   📊 Анализ колокализации...")
            print(f"   📈 Обнаружено везикул: {total_vesicles}")
            print(f"   📈 Обнаружено макрофагов: {np.max(macrophages_labels)}")
            
            # ⭐ Анализ распределения размеров везикул
            if total_vesicles > 0:
                areas = [prop.area for prop in vesicles_props]
                avg_area = np.mean(areas)
                max_area = np.max(areas)
                print(f"   📏 Размеры везикул: средний={avg_area:.1f}, максимальный={max_area}")
            
            for i, vesicle_props in enumerate(vesicles_props):
                vesicle_coords = vesicle_props.coords
                overlapping_macrophages = set()
                
                for coord in vesicle_coords:
                    y, x = coord
                    if y < macrophages_labels.shape[0] and x < macrophages_labels.shape[1]:
                        macrophage_label = macrophages_labels[y, x]
                        if macrophage_label > 0:
                            overlapping_macrophages.add(macrophage_label)
                
                is_colocalized = len(overlapping_macrophages) > 0
                
                if is_colocalized:
                    colocalized_vesicles += 1
                
                colocalization_data.append({
                    'vesicle_id': i + 1,
                    'area': vesicle_props.area,
                    'centroid_x': vesicle_props.centroid[1],
                    'centroid_y': vesicle_props.centroid[0],
                    'colocalized': is_colocalized,
                    'macrophages_count': len(overlapping_macrophages),
                    'size_category': 'large' if vesicle_props.area > self.max_vesicle_size else 'normal'  # ⭐ НОВОЕ ПОЛЕ
                })
            
            percentage = (colocalized_vesicles / total_vesicles * 100) if total_vesicles > 0 else 0
            
            # ⭐ АНАЛИЗ ВЕЗИКУЛ В КЛЕТКАХ (пересечение с ядрами)
            vesicles_in_cells = self.analyze_vesicles_in_cells(vesicles_binary, nuclei_channel, sample_name)
            
            print(f"   ✅ Везикул в макрофагах: {colocalized_vesicles} ({percentage:.1f}%)")
            
            return percentage, colocalized_vesicles, total_vesicles, vesicles_in_cells, colocalization_data
            
        except Exception as e:
            print(f"   ❌ Ошибка анализа колокализации: {e}")
            return 0, 0, 0, 0, []
    
    def process_single_sample(self, sample_data, output_dir, sample_num, total_samples):
        """Обработка одного образца"""
        sample_name = sample_data['sample_name']
        mouse_id = sample_data['mouse_id']
        
        print(f"\n[{sample_num}/{total_samples}] 🔬 ОБРАЗЕЦ: {sample_name}")
        print(f"   📁 Группа: {mouse_id}")
        
        mouse_dir = os.path.join(output_dir, mouse_id)
        os.makedirs(mouse_dir, exist_ok=True)
        
        result = self.create_composite_image(sample_data, mouse_dir)
        if result[0] is None:
            print(f"   ❌ Пропуск образца (ошибка создания композита)")
            return None
        
        # ⭐ ИСПРАВЛЕНИЕ: vesicles_binary уже готовая маска из create_composite_image!
        composite, nuclei_processed, vesicles_binary, protein_processed, composite_filename, vesicles_original = result
        
        # Сегментируем макрофаги
        print(f"   🔍 Сегментация макрофагов...")
        macrophages_labels = self.segment_macrophages(nuclei_processed, protein_processed)
        
        if vesicles_binary is not None and macrophages_labels is not None:
            self.diagnose_colocalization(vesicles_binary, macrophages_labels, nuclei_processed, sample_name)
            # Проверяем, что в маске есть везикулы
            if np.sum(vesicles_binary) == 0:
                print(f"   ⚠️ Маска везикул пуста!")
            else:
                print(f"   ✅ Маска везикул содержит {np.sum(vesicles_binary)} пикселей")
            
            # Анализируем колокализацию
            percentage, colocalized, total, vesicles_in_cells, colocalization_data = self.analyze_colocalization(
                vesicles_binary, macrophages_labels, nuclei_processed, sample_name
            )

            
            safe_name = "".join(c for c in sample_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            csv_filename = f"{safe_name}_results.csv"
            csv_path = os.path.join(mouse_dir, csv_filename)
            
            results_df = pd.DataFrame(colocalization_data)
            results_df.to_csv(csv_path, index=False, encoding='utf-8')
            
            summary = {
                'sample_name': sample_name,
                'mouse_id': mouse_id,
                'composite_image': composite_filename,
                'results_csv': csv_filename,
                'total_vesicles': total,
                'colocalized_vesicles': colocalized,
                'vesicles_in_cells': vesicles_in_cells,
                'colocalization_percentage': percentage,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'has_original_vesicles': vesicles_original is not None
            }
            
            print(f"   ✅ РЕЗУЛЬТАТЫ:")
            print(f"      • Везикул всего: {total}")
            print(f"      • Везикул в макрофагах: {colocalized}")
            print(f"      • Везикул в клетках: {vesicles_in_cells}")
            print(f"      • Процент колокализации: {percentage:.2f}%")
            
            return summary
        else:
            print(f"   ❌ Ошибка сегментации")
            return None
    
    def process_lif_file(self, lif_file_path, output_base_dir):
        """Обработка всего LIF файла"""
        start_time = time.time()
        
        # Создаем уникальную папку для результатов
        base_name = os.path.splitext(os.path.basename(lif_file_path))[0]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(output_base_dir, f"analysis_{base_name}_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n🚀 НАЧАЛО ОБРАБОТКИ LIF ФАЙЛА")
        print(f"📁 Входной файл: {lif_file_path}")
        print(f"📁 Выходная папка: {output_dir}")
        print("=" * 60)
        
        # Загрузка данных из LIF файла
        samples_data = self.load_all_samples_from_lif(lif_file_path)
        
        if not samples_data:
            error_msg = f"❌ ОШИБКА: Не удалось загрузить данные из LIF файла: {lif_file_path}"
            print(error_msg)
            return None, error_msg
        
        total_samples = sum(len(samples) for samples in samples_data.values())
        print(f"\n🎯 ВСЕГО ОБРАЗЦОВ ДЛЯ ОБРАБОТКИ: {total_samples}")
        
        all_summaries = []
        processed_count = 0
        
        for mouse_id, samples in samples_data.items():
            print(f"\n📂 ГРУППА: {mouse_id} ({len(samples)} образцов)")
            
            for sample_data in samples:
                processed_count += 1
                summary = self.process_single_sample(sample_data, output_dir, processed_count, total_samples)
                
                if summary:
                    all_summaries.append(summary)
                
                print("-" * 50)
        
        if all_summaries:
            summary_df = pd.DataFrame(all_summaries)
            summary_path = os.path.join(output_dir, "00_analysis_summary.csv")
            summary_df.to_csv(summary_path, index=False, encoding='utf-8')
            
            total_processed = len(all_summaries)
            avg_percentage = summary_df['colocalization_percentage'].mean()
            processing_time = time.time() - start_time
            
            print(f"\n" + "=" * 60)
            print(f"✅ ОБРАБОТКА ЗАВЕРШЕНА!")
            print(f"📊 ОБРАБОТАНО: {total_processed}/{total_samples} образцов")
            print(f"📈 СРЕДНИЙ ПРОЦЕНТ КОЛОКАЛИЗАЦИИ: {avg_percentage:.2f}%")
            print(f"⏱️ ВРЕМЯ: {processing_time:.2f} сек")
            print(f"💾 РЕЗУЛЬТАТЫ: {output_dir}")
            print("=" * 60)
            
            return all_summaries, output_dir
        else:
            error_msg = "❌ ОШИБКА: Не удалось обработать ни одного образца"
            print(error_msg)
            return None, error_msg
        
    def draw_vesicles_on_channel(self, channel, vesicles_binary, color=(0, 255, 0), thickness=2):
        """Отрисовывает круги вокруг везикул на канале"""
        if channel is None or vesicles_binary is None:
            return channel
            
        try:
            # Конвертируем в цветное изображение если нужно
            if len(channel.shape) == 2:
                colored_channel = cv2.cvtColor(channel, cv2.COLOR_GRAY2BGR)
            else:
                colored_channel = channel.copy()
            
            # Находим центры везикул
            vesicles_labels = measure.label(vesicles_binary)
            vesicles_props = measure.regionprops(vesicles_labels)
            
            for region in vesicles_props:
                # Получаем центр и радиус везикулы
                y, x = region.centroid
                radius = int(np.sqrt(region.area / np.pi)) + 2  # Немного увеличиваем радиус для видимости
                
                # Рисуем круг
                cv2.circle(colored_channel, (int(x), int(y)), radius, color, thickness)
                
            return colored_channel
            
        except Exception as e:
            print(f"   ❌ Ошибка отрисовки везикул: {e}")
            return channel


    def draw_vesicles_and_macrophages_on_composite(self, composite, vesicles_binary, macrophages_labels, 
                                                vesicles_color=(0, 255, 255), macrophages_color=(255, 255, 255), 
                                                thickness=2):
        """Отрисовывает круги вокруг везикул и макрофагов на композите"""
        if composite is None:
            return composite
            
        try:
            composite_with_annotations = composite.copy()
            
            # Рисуем везикулы (желтые круги)
            if vesicles_binary is not None:
                vesicles_labels = measure.label(vesicles_binary)
                vesicles_props = measure.regionprops(vesicles_labels)
                
                for region in vesicles_props:
                    y, x = region.centroid
                    radius = int(np.sqrt(region.area / np.pi)) + 2
                    cv2.circle(composite_with_annotations, (int(x), int(y)), radius, vesicles_color, thickness)
            
            # Рисуем макрофаги (белые круги)
            if macrophages_labels is not None:
                macrophages_props = measure.regionprops(macrophages_labels)
                
                for region in macrophages_props:
                    y, x = region.centroid
                    radius = int(np.sqrt(region.area / np.pi)) + 5
                    cv2.circle(composite_with_annotations, (int(x), int(y)), radius, macrophages_color, thickness)
                    
            return composite_with_annotations
            
        except Exception as e:
            print(f"   ❌ Ошибка отрисовки аннотаций: {e}")
            return composite
        

    def enhance_vesicles_brightness(self, vesicles_channel, enhancement_factor=1.0):
        """МИНИМАЛЬНОЕ улучшение яркости везикул - чтобы не сливались"""
        if vesicles_channel is None:
            return vesicles_channel
        
        try:
            # ⭐ МИНИМАЛЬНАЯ ОБРАБОТКА - только легкое шумоподавление
            # Убираем все агрессивные усиления!
            
            # Легкое размытие для уменьшения шума
            vesicles_enhanced = cv2.medianBlur(vesicles_channel, 3)
            
            # ⭐ ОЧЕНЬ ЛЕГКОЕ увеличение контраста (если нужно)
            if enhancement_factor > 1.0:
                vesicles_enhanced = cv2.convertScaleAbs(vesicles_enhanced, alpha=1.1, beta=5)
            
            print(f"      ✅ Применено легкое улучшение везикул")
            
            return vesicles_enhanced
            
        except Exception as e:
            print(f"      ❌ Ошибка улучшения яркости везикул: {e}")
            return vesicles_channel
            
        
    def aggressive_vesicles_enhancement(self, vesicles_channel):
        """Агрессивное усиление везикул после коррекции фона"""
        if vesicles_channel is None:
            return vesicles_channel
        
        try:
            print("      🔥 АГРЕССИВНОЕ УСИЛЕНИЕ ВЕЗИКУЛ...")
            
            # Анализ текущего состояния
            original_stats = {
                'max': np.max(vesicles_channel),
                'mean': np.mean(vesicles_channel),
                'std': np.std(vesicles_channel)
            }
            print(f"         До усиления: max={original_stats['max']:.1f}, mean={original_stats['mean']:.1f}")
            
            # Метод 1: Gamma коррекция для усиления темных областей
            gamma = 0.5  # <1 для усиления темных тонов
            vesicles_gamma = exposure.adjust_gamma(vesicles_channel, gamma=gamma)
            
            # Метод 2: Логарифмическое преобразование
            vesicles_log = vesicles_channel.astype(np.float32) + 1
            vesicles_log = np.log(vesicles_log)
            vesicles_log = cv2.normalize(vesicles_log, None, 0, 255, cv2.NORM_MINMAX)
            vesicles_log = vesicles_log.astype(np.uint8)
            
            # Метод 3: Умножение интенсивностей
            vesicles_multiplied = vesicles_channel.astype(np.float32) * 3.0
            vesicles_multiplied = np.clip(vesicles_multiplied, 0, 255).astype(np.uint8)
            
            # Комбинируем методы: берем максимум из всех методов
            vesicles_combined = np.maximum.reduce([vesicles_gamma, vesicles_log, vesicles_multiplied])
            
            # Применяем CLAHE для локального контраста
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
            vesicles_enhanced = clahe.apply(vesicles_combined)
            
            # Легкое медианное размытие для уменьшения шума
            vesicles_enhanced = cv2.medianBlur(vesicles_enhanced, 3)
            
            # Финальная нормализация
            vesicles_enhanced = cv2.normalize(vesicles_enhanced, None, 0, 255, cv2.NORM_MINMAX)
            
            enhanced_stats = {
                'max': np.max(vesicles_enhanced),
                'mean': np.mean(vesicles_enhanced),
                'std': np.std(vesicles_enhanced)
            }
            
            print(f"         После усиления: max={enhanced_stats['max']:.1f}, mean={enhanced_stats['mean']:.1f}")
            print(f"         Усиление: {enhanced_stats['mean']/original_stats['mean']:.1f}x")
            
            return vesicles_enhanced.astype(np.uint8)
            
        except Exception as e:
            print(f"         ❌ Ошибка агрессивного усиления: {e}")
            return vesicles_channel
        

    def analyze_vesicles_in_cells(self, vesicles_binary, nuclei_channel, sample_name):
        """Анализ количества везикул внутри клеток (пересечение с ядрами)"""
        if vesicles_binary is None or nuclei_channel is None:
            return 0
        
        try:
            print("   🔍 Анализ везикул в клетках...")
            
            # Сегментируем ядра
            nuclei_threshold = filters.threshold_otsu(nuclei_channel)
            nuclei_binary = nuclei_channel > nuclei_threshold
            
            # Убираем мелкие объекты в ядрах
            nuclei_cleaned = morphology.remove_small_objects(nuclei_binary, min_size=50)
            
            # Находим пересечение везикул с ядрами
            vesicles_in_nuclei = np.logical_and(vesicles_binary > 0, nuclei_cleaned)
            
            # Метка пересекающихся областей
            labeled_intersection = measure.label(vesicles_in_nuclei)
            regions_intersection = measure.regionprops(labeled_intersection)
            
            vesicles_in_cells_count = len(regions_intersection)
            
            # Общее количество везикул для сравнения
            labeled_vesicles = measure.label(vesicles_binary)
            total_vesicles = np.max(labeled_vesicles) if np.max(labeled_vesicles) > 0 else 0
            
            if total_vesicles > 0:
                percentage = (vesicles_in_cells_count / total_vesicles) * 100
                print(f"   ✅ Везикул в клетках: {vesicles_in_cells_count}/{total_vesicles} ({percentage:.1f}%)")
            else:
                print(f"   ✅ Везикул в клетках: {vesicles_in_cells_count}/0")
            
            return vesicles_in_cells_count
            
        except Exception as e:
            print(f"   ❌ Ошибка анализа везикул в клетках: {e}")
            return 0


    def apply_black_background(self, colored_image, gray_channel, threshold=5):
        """Применяет черный фон к цветному изображению на основе порога"""
        if colored_image is None or gray_channel is None:
            return colored_image
        
        # Создаем маску для фона (пиксели ниже порога)
        background_mask = gray_channel <= threshold
        
        # Устанавливаем эти пиксели в черный цвет
        colored_image[background_mask] = [0, 0, 0]
        
        return colored_image
    
    
        

    def find_vesicles_by_intensity_difference(self, vesicles_channel, collagen_channel, sample_name):
        """Находит везикулы по разности интенсивностей - включая тусклые"""
        if vesicles_channel is None or collagen_channel is None:
            return None, None
        
        try:
            print(f"   🔍 Поиск везикул по разности интенсивностей...")
            
            # ⭐ Анализ распределения интенсивностей
            vesicles_stats = {
                'min': np.min(vesicles_channel),
                'max': np.max(vesicles_channel), 
                'mean': np.mean(vesicles_channel),
                'median': np.median(vesicles_channel)
            }
            
            collagen_stats = {
                'min': np.min(collagen_channel),
                'max': np.max(collagen_channel),
                'mean': np.mean(collagen_channel),
                'median': np.median(collagen_channel)
            }
            
            print(f"      📊 Канал везикул: min={vesicles_stats['min']}, max={vesicles_stats['max']}, mean={vesicles_stats['mean']:.1f}, median={vesicles_stats['median']:.1f}")
            print(f"      📊 Канал коллагена: min={collagen_stats['min']}, max={collagen_stats['max']}, mean={collagen_stats['mean']:.1f}, median={collagen_stats['median']:.1f}")
            
            # ⭐ АДАПТИВНЫЕ ПОРОГИ на основе статистики
            vesicles_float = vesicles_channel.astype(np.float32)
            collagen_float = collagen_channel.astype(np.float32)
            
            # Избегаем деления на ноль
            collagen_float[collagen_float == 0] = 1
            
            # ⭐ ГИБКИЕ ПОРОГИ:
            # Отношение интенсивностей
            intensity_ratio = vesicles_float / collagen_float
            ratio_threshold = self.intensity_ratio_threshold
            
            # Абсолютная разность (адаптивная)
            intensity_diff = vesicles_float - collagen_float
            diff_threshold = max(10, collagen_stats['mean'] * 0.3)  # ⭐ АДАПТИВНЫЙ ПОРОГ
            
            # ⭐ ДОПОЛНИТЕЛЬНО: везикулы должны быть достаточно яркими
            vesicles_min_brightness = max(15, vesicles_stats['mean'] * 0.3)  # ⭐ АДАПТИВНЫЙ МИНИМУМ
            
            # Комбинированная маска
            ratio_mask = intensity_ratio > ratio_threshold
            diff_mask = intensity_diff > diff_threshold
            brightness_mask = vesicles_channel > vesicles_min_brightness
            
            candidate_mask = ratio_mask & diff_mask & brightness_mask
            
            print(f"      🎯 Пороги: ratio>{ratio_threshold}, diff>{diff_threshold:.1f}, brightness>{vesicles_min_brightness:.1f}")
            print(f"      📈 Кандидатов в везикулы: {np.sum(candidate_mask)} пикселей")
            
            # Создаем бинарную маску кандидатов
            candidate_binary = candidate_mask.astype(np.uint8) * 255
            
            # Очистка маски от шума
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            candidate_cleaned = cv2.morphologyEx(candidate_binary, cv2.MORPH_OPEN, kernel)
            
            candidate_coords = np.where(candidate_cleaned > 0)
            print(f"      ✅ После очистки: {len(candidate_coords[0])} пикселей-кандидатов")
            
            return candidate_cleaned, candidate_coords
            
        except Exception as e:
            print(f"   ❌ Ошибка поиска везикул по разности: {e}")
            return None, None
        

    def intensity_based_segmentation(self, cluster_mask, vesicles_channel, optimal_count):
        """Сегментация на основе анализа интенсивности"""
        try:
            print("            💡 Используем сегментацию по интенсивности...")
            
            # Создаем маску только для области скопления
            cluster_region = np.zeros_like(vesicles_channel, dtype=np.float32)
            cluster_region[cluster_mask] = vesicles_channel[cluster_mask]
            
            # Нормализуем интенсивность в скоплении
            if np.max(cluster_region) > 0:
                cluster_region = cluster_region / np.max(cluster_region)
            
            # Применяем watershed на основе градиента интенсивности
            from skimage.feature import peak_local_max
            
            # Расстояние + интенсивность
            distance = ndimage.distance_transform_edt(cluster_mask)
            weighted_map = distance * cluster_region
            
            # Находим локальные максимумы
            coordinates = peak_local_max(
                weighted_map, 
                min_distance=5,
                num_peaks=optimal_count * 2,
                threshold_abs=np.percentile(weighted_map[cluster_mask], 40)
            )
            
            if len(coordinates) > 0:
                # Создаем маркеры
                markers = np.zeros_like(vesicles_channel, dtype=np.int32)
                for i, (y, x) in enumerate(coordinates):
                    markers[y, x] = i + 1
                
                # Watershed
                labels = segmentation.watershed(
                    -weighted_map, 
                    markers, 
                    mask=cluster_mask,
                    watershed_line=True
                )
                
                # Создаем маску результатов
                result_mask = np.zeros_like(cluster_mask, dtype=bool)
                for label in range(1, np.max(labels) + 1):
                    vesicle_mask = labels == label
                    if np.sum(vesicle_mask) >= self.min_vesicle_size:
                        result_mask[vesicle_mask] = True
                
                return result_mask
            else:
                print("            ⚠️  Не найдено максимумов, используем равномерную сетку")
                return self.uniform_grid_segmentation(cluster_mask, np.sum(cluster_mask), 
                                                    (self.min_vesicle_size + self.max_vesicle_size) // 2)
                
        except Exception as e:
            print(f"            ❌ Ошибка сегментации по интенсивности: {e}")
            return self.uniform_grid_segmentation(cluster_mask, np.sum(cluster_mask), 
                                                (self.min_vesicle_size + self.max_vesicle_size) // 2)
        

    def apply_colored_channel_simple(self, gray_channel, color):
        """Простое окрашивание канала БЕЗ удаления фона"""
        if gray_channel is None:
            return None
            
        height, width = gray_channel.shape
        colored = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Простое окрашивание без удаления фона
        if color == 'red':
            colored[:,:,2] = gray_channel
        elif color == 'green':
            colored[:,:,1] = gray_channel
        elif color == 'blue':
            colored[:,:,0] = gray_channel
        elif color == 'yellow':
            colored[:,:,1] = gray_channel
            colored[:,:,2] = gray_channel
        
        return colored
    

    def subtract_vesicles_background_local(self, vesicles_channel, collagen_channel, sample_name):
        """Локальное вычитание фона везикул: вычитаем только там, где коллаген ярче везикул"""
        if vesicles_channel is None or collagen_channel is None:
            return vesicles_channel
        
        try:
            print(f"\n   🔍 ЛОКАЛЬНОЕ ВЫЧИТАНИЕ ДЛЯ: {sample_name}")
            
            # Конвертируем в float для точных вычислений
            vesicles_float = vesicles_channel.astype(np.float32)
            collagen_float = collagen_channel.astype(np.float32)
            
            # Статистика до вычитания
            vesicles_before = {
                'mean': np.mean(vesicles_float),
                'median': np.median(vesicles_float),
                'max': np.max(vesicles_float)
            }
            
            collagen_stats = {
                'mean': np.mean(collagen_float),
                'median': np.median(collagen_float),
                'max': np.max(collagen_float)
            }
            
            print(f"      📊 Везикулы ДО: mean={vesicles_before['mean']:.2f}, median={vesicles_before['median']:.2f}")
            print(f"      📊 Коллаген: mean={collagen_stats['mean']:.2f}, median={collagen_stats['median']:.2f}")
            
            # 1. Находим пиксели, где коллаген ярче везикул
            collagen_brighter_mask = collagen_float > vesicles_float
            brighter_count = np.sum(collagen_brighter_mask)
            total_pixels = vesicles_float.size
            
            print(f"      🔍 Анализ яркости:")
            print(f"         • Всего пикселей: {total_pixels}")
            print(f"         • Коллаген ярче везикул: {brighter_count} ({brighter_count/total_pixels*100:.1f}%)")
            
            if brighter_count == 0:
                print("      ✅ Коллаген НИГДЕ не ярче везикул, вычитание не требуется")
                return vesicles_channel
            
            # 2. В этих пикселях вычисляем разницу
            difference = collagen_float[collagen_brighter_mask] - vesicles_float[collagen_brighter_mask]
            
            # Статистика разницы
            if len(difference) > 0:
                print(f"      📈 Разница (коллаген - везикулы):")
                print(f"         • Средняя: {np.mean(difference):.2f}")
                print(f"         • Медиана: {np.median(difference):.2f}")
                print(f"         • Максимальная: {np.max(difference):.2f}")
                print(f"         • Минимальная: {np.min(difference):.2f}")
            
            # 3. Стратегии вычитания:
            print(f"\n      🎯 СТРАТЕГИИ ВЫЧИТАНИЯ:")
            
            # Создаем копию для результата
            result = vesicles_float.copy()
            
            # Вариант A: Вычитаем полную разницу там, где коллаген ярче
            if False:  # Измените на True для выбора этой стратегии
                print(f"      🅰️  Вычитаем ПОЛНУЮ РАЗНИЦУ")
                result[collagen_brighter_mask] -= difference
                subtraction_type = "full_difference"
            
            # Вариант B: Вычитаем процент от разницы (например, 50%)
            elif False:  # Измените на True для выбора этой стратегии
                percent_to_subtract = 0.5  # 50% от разницы
                print(f"      🅱️  Вычитаем {percent_to_subtract*100:.0f}% от разницы")
                result[collagen_brighter_mask] -= difference * percent_to_subtract
                subtraction_type = f"{percent_to_subtract*100:.0f}%_of_difference"
            
            # Вариант C: Вычитаем фиксированное значение на основе статистики (РЕКОМЕНДУЕМЫЙ)
            else:
                # Используем медиану разницы, но ограничиваем, чтобы не вычесть слишком много
                median_diff = np.median(difference) if len(difference) > 0 else 0
                
                # Вычисляем безопасное значение для вычитания
                safe_subtraction = min(median_diff * 0.7, np.percentile(vesicles_float, 30))
                safe_subtraction = max(0, safe_subtraction)  # Не отрицательное
                
                print(f"      🅲️  Вычитаем БЕЗОПАСНОЕ значение: {safe_subtraction:.2f}")
                print(f"         (рассчитано как 70% от медианной разницы)")
                
                result[collagen_brighter_mask] -= safe_subtraction
                subtraction_type = f"safe_{safe_subtraction:.1f}"
            
            # 4. Обрезаем отрицательные значения
            result = np.clip(result, 0, 255)
            
            # 5. Анализ результатов
            vesicles_after = {
                'mean': np.mean(result),
                'median': np.median(result),
                'max': np.max(result)
            }
            
            print(f"\n      📊 РЕЗУЛЬТАТЫ ВЫЧИТАНИЯ:")
            print(f"         • Средняя яркость: {vesicles_before['mean']:.2f} → {vesicles_after['mean']:.2f}")
            print(f"         • Медиана: {vesicles_before['median']:.2f} → {vesicles_after['median']:.2f}")
            print(f"         • Изменение: -{(vesicles_before['mean'] - vesicles_after['mean']):.2f} единиц")
            
            if vesicles_before['mean'] > 0:
                reduction_percent = (vesicles_before['mean'] - vesicles_after['mean']) / vesicles_before['mean'] * 100
                print(f"         • Процент снижения: {reduction_percent:.1f}%")
            
            # 6. Дополнительная диагностика: сколько пикселей стало нулевыми
            became_zero = np.sum((vesicles_float > 0) & (result == 0))
            if became_zero > 0:
                print(f"         ⚠️  Пикселей стало нулевыми: {became_zero} ({became_zero/total_pixels*100:.2f}%)")
            
            # 7. Визуализация для отладки (опционально)
            if brighter_count > 0 and brighter_count < total_pixels * 0.3:  # Если не слишком много
                self._visualize_subtraction(vesicles_channel, collagen_channel, result, 
                                        collagen_brighter_mask, sample_name, mouse_dir)
            
            # Конвертируем обратно в uint8
            result_uint8 = result.astype(np.uint8)
            
            return result_uint8
            
        except Exception as e:
            print(f"      ❌ Ошибка локального вычитания: {e}")
            import traceback
            traceback.print_exc()
            return vesicles_channel

    def _visualize_subtraction(self, vesicles_orig, collagen, vesicles_result, mask, sample_name, output_dir):
        """Визуализация процесса вычитания для отладки"""
        try:
            safe_name = "".join(c for c in sample_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            
            # Создаем визуализацию
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # 1. Оригинальные везикулы
            axes[0, 0].imshow(vesicles_orig, cmap='hot')
            axes[0, 0].set_title('Везикулы (оригинал)')
            axes[0, 0].axis('off')
            
            # 2. Коллаген
            axes[0, 1].imshow(collagen, cmap='gray')
            axes[0, 1].set_title('Коллаген')
            axes[0, 1].axis('off')
            
            # 3. Маска: где коллаген ярче
            axes[0, 2].imshow(mask, cmap='binary')
            axes[0, 2].set_title(f'Коллаген ярче ({np.sum(mask)}px)')
            axes[0, 2].axis('off')
            
            # 4. Результат вычитания
            axes[1, 0].imshow(vesicles_result, cmap='hot')
            axes[1, 0].set_title('Везикулы (после вычитания)')
            axes[1, 0].axis('off')
            
            # 5. Разница
            diff = vesicles_orig.astype(np.float32) - vesicles_result.astype(np.float32)
            axes[1, 1].imshow(diff, cmap='coolwarm', vmin=-50, vmax=50)
            axes[1, 1].set_title('Вычтено (разница)')
            axes[1, 1].axis('off')
            
            # 6. Гистограмма изменений
            diff_flat = diff.flatten()
            axes[1, 2].hist(diff_flat[diff_flat != 0], bins=50, color='blue', alpha=0.7)
            axes[1, 2].set_title('Распределение вычтенных значений')
            axes[1, 2].set_xlabel('Вычтено единиц')
            axes[1, 2].set_ylabel('Частота')
            
            plt.suptitle(f'Локальное вычитание: {sample_name}', fontsize=16)
            plt.tight_layout()
            
            # Сохраняем
            viz_path = os.path.join(output_dir, f"{safe_name}_subtraction_analysis.png")
            plt.savefig(viz_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"      📸 Визуализация сохранена: {viz_path}")
            
        except Exception as e:
            print(f"      ⚠️  Ошибка визуализации: {e}")


    def watershed_intensity_segmentation(self, cluster_mask, vesicles_channel, optimal_count):
        """Watershed сегментация на основе градиента интенсивности"""
        try:
            # Создаем маску только для области скопления
            cluster_region = np.zeros_like(vesicles_channel, dtype=np.float32)
            cluster_region[cluster_mask] = vesicles_channel[cluster_mask]
            
            # Нормализуем
            if np.max(cluster_region) > 0:
                cluster_region = cluster_region / np.max(cluster_region)
            
            # Расстояние + интенсивность
            distance = ndimage.distance_transform_edt(cluster_mask)
            
            # Комбинируем расстояние и интенсивность
            # Везикулы обычно ярче по краям
            gradient = filters.sobel(cluster_region)
            weighted_map = distance * (1.0 - gradient)  # Минимумы в ярких областях
            
            # Находим маркеры - локальные минимумы (для watershed нужны минимумы)
            from skimage.feature import peak_local_max
            
            # Инвертируем для поиска минимумов
            inverted_map = -weighted_map
            
            coordinates = peak_local_max(
                inverted_map, 
                min_distance=5,
                num_peaks=optimal_count * 2,  # Ищем больше пиков
                threshold_abs=np.percentile(inverted_map[cluster_mask], 30),
                footprint=np.ones((3, 3))
            )
            
            if len(coordinates) < 2:  # Если не нашли достаточно пиков
                print("            ⚠️  Не найдено достаточно маркеров")
                return np.zeros_like(cluster_mask, dtype=bool)
            
            # Создаем маркеры
            markers = np.zeros_like(vesicles_channel, dtype=np.int32)
            for i, (y, x) in enumerate(coordinates):
                if cluster_mask[y, x]:  # Только внутри маски
                    markers[y, x] = i + 1
            
            # Watershed
            labels = segmentation.watershed(
                weighted_map, 
                markers, 
                mask=cluster_mask,
                watershed_line=True,
                compactness=0.01
            )
            
            # Создаем маску результатов
            result_mask = np.zeros_like(cluster_mask, dtype=bool)
            for label in range(1, np.max(labels) + 1):
                vesicle_mask = labels == label
                area = np.sum(vesicle_mask)
                
                # Фильтруем по размеру
                if self.min_vesicle_size <= area <= self.max_vesicle_size * 1.5:
                    result_mask[vesicle_mask] = True
            
            return result_mask
            
        except Exception as e:
            print(f"            ❌ Ошибка watershed сегментации: {e}")
            return np.zeros_like(cluster_mask, dtype=bool)
        

    def uniform_grid_segmentation_improved(self, cluster_mask, cluster_area, target_size, y_min, y_max, x_min, x_max):
        """Улучшенная равномерная сетка с проверкой интенсивности"""
        try:
            height, width = y_max - y_min + 1, x_max - x_min + 1
            
            # Оптимальное количество везикул
            optimal_count = max(2, int(cluster_area / target_size))
            
            # Определяем размер сетки
            grid_size = int(np.ceil(np.sqrt(optimal_count * 1.2)))  # +20% для перекрытия
            
            # Размер ячейки
            cell_height = max(5, height // grid_size)
            cell_width = max(5, width // grid_size)
            
            print(f"            📐 Равномерная сетка: {grid_size}x{grid_size}, ячейка: {cell_width}x{cell_height}")
            
            result_mask = np.zeros_like(cluster_mask, dtype=bool)
            
            for i in range(grid_size):
                for j in range(grid_size):
                    # Центр ячейки
                    y_center = y_min + (i + 0.5) * cell_height
                    x_center = x_min + (j + 0.5) * cell_width
                    
                    y_center_int = int(round(y_center))
                    x_center_int = int(round(x_center))
                    
                    # Проверяем, что центр внутри маски
                    if (0 <= y_center_int < cluster_mask.shape[0] and 
                        0 <= x_center_int < cluster_mask.shape[1] and 
                        cluster_mask[y_center_int, x_center_int]):
                        
                        # Проверяем интенсивность в центре
                        center_intensity = vesicles_channel[y_center_int, x_center_int]
                        if center_intensity < self.min_vesicle_intensity:
                            continue  # Пропускаем тусклые центры
                        
                        # Радиус везикулы (половина от среднего размера)
                        radius = int(np.sqrt(target_size / np.pi))
                        
                        # Создаем круглую маску
                        y_start = max(0, y_center_int - radius)
                        y_end = min(cluster_mask.shape[0], y_center_int + radius + 1)
                        x_start = max(0, x_center_int - radius)
                        x_end = min(cluster_mask.shape[1], x_center_int + radius + 1)
                        
                        if y_end > y_start and x_end > x_start:
                            # Круг в локальных координатах
                            yy_local, xx_local = np.ogrid[y_start:y_end, x_start:x_end]
                            circle_mask = (xx_local - x_center_int)**2 + (yy_local - y_center_int)**2 <= radius**2
                            
                            # Пересекаем с кластером
                            cluster_slice = cluster_mask[y_start:y_end, x_start:x_end]
                            final_circle = circle_mask & cluster_slice
                            
                            # Проверяем среднюю интенсивность кандидата
                            if np.sum(final_circle) >= self.min_vesicle_size:
                                candidate_intensities = vesicles_channel[y_start:y_end, x_start:x_end][final_circle]
                                if np.mean(candidate_intensities) >= self.min_vesicle_intensity:
                                    result_mask[y_start:y_end, x_start:x_end] |= final_circle
            
            return result_mask
            
        except Exception as e:
            print(f"            ❌ Ошибка улучшенной сетки: {e}")
            return self.uniform_grid_segmentation(cluster_mask, cluster_area, target_size)

    def finalize_vesicles(self, vesicles_mask, target_size):
        """Финальная обработка и очистка масок везикул"""
        try:
            # 1. Удаляем слишком мелкие объекты
            cleaned = morphology.remove_small_objects(vesicles_mask, min_size=self.min_vesicle_size)
            
            # 2. Разделяем слабо соединенные области
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            opened = cv2.morphologyEx(cleaned.astype(np.uint8), cv2.MORPH_OPEN, kernel_open)
            
            # 3. Удаляем слишком крупные объекты (если остались)
            labels = measure.label(opened)
            regions = measure.regionprops(labels)
            
            final_mask = np.zeros_like(vesicles_mask, dtype=bool)
            
            for region in regions:
                if region.area <= self.max_vesicle_size * 1.2:  # Допуск 20%
                    final_mask[labels == region.label] = True
                else:
                    # Если объект все еще слишком большой, принудительно делим
                    print(f"               ⚠️  Объект все еще слишком большой: {region.area}px")
                    
                    # Берем bounding box и делим на 4 части
                    y0, x0, y1, x1 = region.bbox
                    sub_height = (y1 - y0) // 2
                    sub_width = (x1 - x0) // 2
                    
                    for i in range(2):
                        for j in range(2):
                            sy0 = y0 + i * sub_height
                            sx0 = x0 + j * sub_width
                            sy1 = min(y1, sy0 + sub_height)
                            sx1 = min(x1, sx0 + sub_width)
                            
                            # Создаем круг в центре подобласти
                            cy = (sy0 + sy1) // 2
                            cx = (sx0 + sx1) // 2
                            radius = int(np.sqrt(target_size / np.pi) * 0.7)  # Немного меньше
                            
                            # Добавляем круг если попадает в маску
                            yy, xx = np.ogrid[sy0:sy1, sx0:sx1]
                            circle = (xx - cx)**2 + (yy - cy)**2 <= radius**2
                            
                            # Пересекаем с оригинальной маской региона
                            region_slice = (labels[sy0:sy1, sx0:sx1] == region.label)
                            final_circle = circle & region_slice
                            
                            if np.sum(final_circle) >= self.min_vesicle_size:
                                final_mask[sy0:sy1, sx0:sx1] |= final_circle
            
            return final_mask
            
        except Exception as e:
            print(f"            ❌ Ошибка финальной обработки: {e}")
            return vesicles_mask
        


    def subtract_vesicles_background_auto(self, vesicles_channel, collagen_channel, sample_name):
        """АВТОМАТИЧЕСКОЕ вычитание фона везикул (без интерактивности)"""
        if vesicles_channel is None:
            return vesicles_channel
        
        try:
            # Используем collagen_percentile вместо жесткого 90
            collagen_percentile_value = np.percentile(collagen_channel, self.collagen_percentile) if collagen_channel is not None else 0
            
            # Анализируем распределение интенсивностей везикул
            vesicles_max = np.max(vesicles_channel)
            vesicles_mean = np.mean(vesicles_channel)
            
            print(f"\n   🔍 АВТОМАТИЧЕСКАЯ КОРРЕКЦИЯ ВЕЗИКУЛ ДЛЯ: {sample_name}")
            print(f"      📊 Коллаген ({self.collagen_percentile}%): {collagen_percentile_value:.2f}")
            print(f"      📈 Везикулы - max: {vesicles_max:.2f}, mean: {vesicles_mean:.2f}")
            
            # Используем subtraction_factor из параметров (например 0.9 = 90%)
            amount_to_subtract = collagen_percentile_value * self.subtraction_factor
            
            print(f"      🧮 Вычитание: {collagen_percentile_value:.2f} × {self.subtraction_factor*100:.1f}% = {amount_to_subtract:.2f}")
            
            # Вычитаем и обрезаем отрицательные значения
            result = vesicles_channel.astype(np.float32) - amount_to_subtract
            result = np.clip(result, 0, 255).astype(np.uint8)
            
            after_max = np.max(result)
            after_mean = np.mean(result)
            print(f"      ✅ Везикулы после: max={after_max:.2f}, mean={after_mean:.2f}")
            
            return result
            
        except Exception as e:
            print(f"      ❌ Ошибка автоматического вычитания: {e}")
            return vesicles_channel
        



    def simple_background_removal(self, vesicles_channel, collagen_channel, sample_name):
        """
        Просто обнуляем пиксели, где коллаген ярче везикул.
        Везикулы остаются без изменений.
        """
        if vesicles_channel is None or collagen_channel is None:
            return vesicles_channel
        
        try:
            print(f"\n   🧹 ПРОСТАЯ ОЧИСТКА ФОНА ДЛЯ: {sample_name}")
            
            # 1. Маска: где коллаген ярче везикул = фон
            background_mask = collagen_channel > vesicles_channel
            
            # 2. Копируем оригинал
            result = vesicles_channel.copy()
            
            # 3. Там, где фон — устанавливаем в 0 (или очень низкое значение)
            result[background_mask] = 0
            
            # 4. Статистика
            affected_pixels = np.sum(background_mask)
            total_pixels = vesicles_channel.size
            
            print(f"      📊 Результат:")
            print(f"         • Обнулено пикселей: {affected_pixels}/{total_pixels} ({affected_pixels/total_pixels*100:.1f}%)")
            print(f"         • Средняя яркость ДО: {np.mean(vesicles_channel):.2f}")
            print(f"         • Средняя яркость ПОСЛЕ: {np.mean(result):.2f}")
            print(f"         • Везикулы сохранены без изменений!")
            
            return result
            
        except Exception as e:
            print(f"      ❌ Ошибка простой очистки: {e}")
            return vesicles_channel
        



    def segment_vesicles_with_criteria(self, vesicles_channel, collagen_channel, sample_name):
        """
        Сегментация везикул с учетом:
        1. Минимальной интенсивности (min_vesicle_intensity)
        2. Превышения над коллагеном (intensity_diff_threshold)
        
        Returns:
            vesicles_binary: бинарная маска везикул
            vesicle_stats: список статистики по везикулам
        """
        if vesicles_channel is None:
            return None, []
        
        try:
            print(f"      🔬 Сегментация с критериями:")
            print(f"         • Мин. интенсивность: {self.min_vesicle_intensity}")
            print(f"         • Мин. разница с коллагеном: {self.intensity_diff_threshold}")
            
            # 1. Базовые маски
            # Маска по минимальной интенсивности
            intensity_mask = vesicles_channel > self.min_vesicle_intensity
            
            # Маска по разнице с коллагеном (если есть)
            if collagen_channel is not None:
                diff_mask = (vesicles_channel.astype(np.float32) - 
                            collagen_channel.astype(np.float32)) > self.intensity_diff_threshold
            else:
                diff_mask = np.ones_like(vesicles_channel, dtype=bool)
            
            # Комбинированная маска
            combined_mask = intensity_mask & diff_mask
            
            print(f"      📊 Статистика масок:")
            print(f"         • По интенсивности (>={self.min_vesicle_intensity}): {np.sum(intensity_mask)} пикселей")
            print(f"         • По разнице с коллагеном: {np.sum(diff_mask)} пикселей")
            print(f"         • Комбинированная: {np.sum(combined_mask)} пикселей")
            
            if np.sum(combined_mask) == 0:
                print(f"      ⚠️  Нет пикселей, удовлетворяющих критериям")
                return np.zeros_like(vesicles_channel, dtype=np.uint8), []
            
            # 2. Сегментация везикул (используем существующий метод)
            # Но передаем комбинированную маску как начальное приближение
            vesicles_binary = self.segment_vesicles_from_mask(vesicles_channel, combined_mask)
            
            # 3. Сбор статистики по найденным везикулам
            if vesicles_binary is not None and np.sum(vesicles_binary) > 0:
                labels = measure.label(vesicles_binary)
                regions = measure.regionprops(labels, intensity_image=vesicles_channel)
                
                vesicle_stats = []
                for i, region in enumerate(regions):
                    # Средняя интенсивность везикулы
                    mean_intensity = region.mean_intensity
                    
                    # Средняя интенсивность коллагена в области везикулы (если есть)
                    if collagen_channel is not None:
                        collagen_in_vesicle = np.mean(collagen_channel[labels == region.label])
                        diff_from_collagen = mean_intensity - collagen_in_vesicle
                    else:
                        collagen_in_vesicle = 0
                        diff_from_collagen = 0
                    
                    vesicle_stats.append({
                        'id': i + 1,
                        'area': region.area,
                        'mean_intensity': mean_intensity,
                        'max_intensity': np.max(vesicles_channel[labels == region.label]),
                        'collagen_in_vesicle': collagen_in_vesicle,
                        'diff_from_collagen': diff_from_collagen,
                        'centroid_y': region.centroid[0],
                        'centroid_x': region.centroid[1]
                    })
                
                print(f"      ✅ Найдено везикул: {len(vesicle_stats)}")
                
                # Дополнительная статистика
                if vesicle_stats:
                    intensities = [v['mean_intensity'] for v in vesicle_stats]
                    print(f"         • Диапазон интенсивностей: {min(intensities):.1f} - {max(intensities):.1f}")
                    print(f"         • Средняя интенсивность: {np.mean(intensities):.1f}")
                
                return vesicles_binary, vesicle_stats
            else:
                print(f"      ⚠️  Не удалось сегментировать везикулы")
                return np.zeros_like(vesicles_channel, dtype=np.uint8), []
            
        except Exception as e:
            print(f"      ❌ Ошибка сегментации с критериями: {e}")
            import traceback
            traceback.print_exc()
            return None, []
        
    def segment_vesicles_from_mask(self, vesicles_channel, initial_mask):
        """
        Сегментирует везикулы, используя начальную маску как основу
        """
        if vesicles_channel is None or initial_mask is None:
            return None
        
        try:
            # Морфологическая очистка начальной маски
            cleaned = morphology.remove_small_objects(initial_mask, min_size=self.min_vesicle_size)
            cleaned = morphology.remove_small_holes(cleaned, area_threshold=5)
            
            # Заполняем дыры
            filled = ndimage.binary_fill_holes(cleaned)
            
            # Морфологическое закрытие для объединения близких пикселей
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            closed = cv2.morphologyEx(filled.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
            
            # Watershed для разделения соединенных объектов
            # Вычисляем расстояние до фона
            distance = ndimage.distance_transform_edt(closed)
            
            # Находим локальные максимумы
            from skimage.feature import peak_local_max
            coordinates = peak_local_max(
                distance, 
                min_distance=5,
                exclude_border=False,
                num_peaks=500
            )
            
            if len(coordinates) > 0:
                # Создаем маркеры
                markers = np.zeros_like(distance, dtype=np.int32)
                for i, (y, x) in enumerate(coordinates):
                    if closed[y, x]:
                        markers[y, x] = i + 1
                
                # Watershed
                labels = segmentation.watershed(-distance, markers, mask=closed)
                
                # Собираем результат
                result_mask = np.zeros_like(closed, dtype=bool)
                for label in range(1, np.max(labels) + 1):
                    vesicle_mask = labels == label
                    area = np.sum(vesicle_mask)
                    
                    # Фильтруем по размеру
                    if self.min_vesicle_size <= area <= self.max_vesicle_size * 2:
                        result_mask[vesicle_mask] = True
            else:
                # Если нет локальных максимумов, используем просто закрытую маску
                result_mask = closed.astype(bool)
            
            # Финальная очистка
            result_mask = morphology.remove_small_objects(result_mask, min_size=self.min_vesicle_size)
            
            return result_mask.astype(np.uint8) * 255
            
        except Exception as e:
            print(f"      ❌ Ошибка сегментации из маски: {e}")
            return initial_mask.astype(np.uint8) * 255
        


    def save_original_channels(self, vesicles_original, protein_original, nuclei_original, collagen_original, safe_name, output_dir):
        """Сохраняет оригинальные каналы - каждый в свой цвет"""
        try:
            # Везикулы (КРАСНЫЙ) - только красный канал
            vesicles_orig_colored = np.zeros((*vesicles_original.shape, 3), dtype=np.uint8)
            vesicles_orig_colored[:,:,2] = vesicles_original  # Только красный
            vesicles_orig_colored = self.add_scale_bar(vesicles_orig_colored, scale_length_pixels=100, scale_text="100 μm")
            cv2.imwrite(os.path.join(output_dir, f"{safe_name}_channel2_vesicles_ORIGINAL.png"), vesicles_orig_colored)
            
            # Белок (ЗЕЛЕНЫЙ) - только зеленый канал
            protein_orig_colored = np.zeros((*protein_original.shape, 3), dtype=np.uint8)
            protein_orig_colored[:,:,1] = protein_original  # Только зеленый
            protein_orig_colored = self.add_scale_bar(protein_orig_colored, scale_length_pixels=100, scale_text="100 μm")
            cv2.imwrite(os.path.join(output_dir, f"{safe_name}_channel3_protein_ORIGINAL.png"), protein_orig_colored)
            
            # Ядра (СИНИЙ) - только синий канал
            nuclei_orig_colored = np.zeros((*nuclei_original.shape, 3), dtype=np.uint8)
            nuclei_orig_colored[:,:,0] = nuclei_original  # Только синий
            nuclei_orig_colored = self.add_scale_bar(nuclei_orig_colored, scale_length_pixels=100, scale_text="100 μm")
            cv2.imwrite(os.path.join(output_dir, f"{safe_name}_channel0_nuclei_ORIGINAL.png"), nuclei_orig_colored)
            
            # Коллаген (можно сохранять как зеленый или отдельно)
            if collagen_original is not None:
                collagen_orig_colored = np.zeros((*collagen_original.shape, 3), dtype=np.uint8)
                collagen_orig_colored[:,:,1] = collagen_original  # Тоже зеленый
                collagen_orig_colored = self.add_scale_bar(collagen_orig_colored, scale_length_pixels=100, scale_text="100 μm")
                cv2.imwrite(os.path.join(output_dir, f"{safe_name}_channel1_collagen_ORIGINAL.png"), collagen_orig_colored)
            
            print(f"      ✅ Оригинальные каналы сохранены")
        except Exception as e:
            print(f"      ⚠️ Ошибка сохранения: {e}")

    def save_processed_channels(self, vesicles_processed, protein_processed, nuclei_processed, composite, safe_name, output_dir):
        """Сохраняет обработанные каналы - каждый в свой цвет"""
        try:
            # Композит (уже цветной)
            composite_with_scale = self.add_scale_bar(composite, scale_length_pixels=100, scale_text="100 μm")
            cv2.imwrite(os.path.join(output_dir, f"{safe_name}_composite.png"), composite_with_scale)
            
            # Везикулы (КРАСНЫЙ)
            vesicles_proc_colored = np.zeros((*vesicles_processed.shape, 3), dtype=np.uint8)
            vesicles_proc_colored[:,:,2] = vesicles_processed
            vesicles_proc_colored = self.add_scale_bar(vesicles_proc_colored, scale_length_pixels=100, scale_text="100 μm")
            cv2.imwrite(os.path.join(output_dir, f"{safe_name}_channel2_vesicles_CORRECTED.png"), vesicles_proc_colored)
            
            # Белок (ЗЕЛЕНЫЙ)
            protein_proc_colored = np.zeros((*protein_processed.shape, 3), dtype=np.uint8)
            protein_proc_colored[:,:,1] = protein_processed
            protein_proc_colored = self.add_scale_bar(protein_proc_colored, scale_length_pixels=100, scale_text="100 μm")
            cv2.imwrite(os.path.join(output_dir, f"{safe_name}_channel3_protein_CORRECTED.png"), protein_proc_colored)
            
            # Ядра (СИНИЙ)
            nuclei_proc_colored = np.zeros((*nuclei_processed.shape, 3), dtype=np.uint8)
            nuclei_proc_colored[:,:,0] = nuclei_processed
            nuclei_proc_colored = self.add_scale_bar(nuclei_proc_colored, scale_length_pixels=100, scale_text="100 μm")
            cv2.imwrite(os.path.join(output_dir, f"{safe_name}_channel0_nuclei_CORRECTED.png"), nuclei_proc_colored)
            
            print(f"      ✅ Обработанные каналы сохранены")
        except Exception as e:
            print(f"      ⚠️ Ошибка сохранения: {e}")

    def save_annotated_images(self, composite, vesicles_mask, macrophages_labels, vesicles_channel, safe_name, output_dir):
        """Сохраняет аннотированные изображения"""
        try:
            if vesicles_mask is not None and np.sum(vesicles_mask) > 0:
                # Композит с аннотациями
                composite_annotated = self.draw_vesicles_and_macrophages_on_composite(
                    composite, vesicles_mask, macrophages_labels,
                    vesicles_color=(0, 255, 255), macrophages_color=(255, 255, 255), thickness=2
                )
                composite_annotated = self.add_scale_bar(composite_annotated, scale_length_pixels=100, scale_text="100 μm")
                cv2.imwrite(os.path.join(output_dir, f"{safe_name}_composite_annotated.png"), composite_annotated)
                
                # Канал везикул с аннотациями
                vesicles_colored = np.zeros((*vesicles_channel.shape, 3), dtype=np.uint8)
                vesicles_colored[:,:,2] = vesicles_channel
                vesicles_annotated = self.draw_vesicles_on_channel(
                    vesicles_colored, vesicles_mask, color=(0, 255, 0), thickness=2
                )
                vesicles_annotated = self.add_scale_bar(vesicles_annotated, scale_length_pixels=100, scale_text="100 μm")
                cv2.imwrite(os.path.join(output_dir, f"{safe_name}_channel2_vesicles_annotated.png"), vesicles_annotated)
                
                print(f"      ✅ Аннотированные изображения сохранены")
        except Exception as e:
            print(f"      ⚠️ Ошибка сохранения аннотаций: {e}")




    def diagnose_colocalization(self, vesicles_binary, macrophages_labels, nuclei_channel, sample_name):
        """Диагностика причин малой колокализации"""
        try:
            print(f"\n   🔬 ДИАГНОСТИКА КОЛОКАЛИЗАЦИИ ДЛЯ: {sample_name}")
            
            # 1. Проверяем маску везикул
            vesicle_pixels = np.sum(vesicles_binary > 0)
            print(f"   📊 Везикулы:")
            print(f"      • Пикселей в маске: {vesicle_pixels}")
            
            if vesicle_pixels == 0:
                print(f"      ⚠️ НЕТ ВЕЗИКУЛ! Проверьте параметры сегментации")
                return
            
            # 2. Проверяем макрофаги
            if macrophages_labels is None:
                print(f"      ⚠️ МАКРОФАГИ НЕ СЕГМЕНТИРОВАНЫ!")
                return
                
            macrophage_pixels = np.sum(macrophages_labels > 0)
            n_macrophages = np.max(macrophages_labels)
            print(f"   📊 Макрофаги:")
            print(f"      • Количество: {n_macrophages}")
            print(f"      • Пикселей в маске: {macrophage_pixels}")
            
            if macrophage_pixels == 0:
                print(f"      ⚠️ НЕТ МАКРОФАГОВ! Проверьте сегментацию макрофагов")
                return
            
            # 3. Проверяем пересечение
            intersection = np.sum((vesicles_binary > 0) & (macrophages_labels > 0))
            print(f"   📊 Пересечение:")
            print(f"      • Пикселей пересечения: {intersection}")
            print(f"      • % везикул в макрофагах: {intersection/vesicle_pixels*100:.2f}%")
            
            if intersection == 0:
                print(f"      ⚠️ НЕТ ПЕРЕСЕЧЕНИЯ!")
                
                # Проверяем, есть ли везикулы рядом с макрофагами
                from scipy import ndimage
                distance_to_macrophages = ndimage.distance_transform_edt(~(macrophages_labels > 0))
                vesicle_distances = distance_to_macrophages[vesicles_binary > 0]
                
                if len(vesicle_distances) > 0:
                    min_dist = np.min(vesicle_distances)
                    mean_dist = np.mean(vesicle_distances)
                    print(f"      • Ближайшая везикула к макрофагу: {min_dist:.1f} пикселей")
                    print(f"      • Среднее расстояние до макрофагов: {mean_dist:.1f} пикселей")
                    
                    if min_dist < 10:
                        print(f"      💡 Везикулы ОЧЕНЬ БЛИЗКО к макрофагам, но не пересекаются!")
                        print(f"         Возможно, нужно расширить маску макрофагов")
            
            # 4. Проверяем координаты
            vesicle_coords = np.where(vesicles_binary > 0)
            macrophage_coords = np.where(macrophages_labels > 0)
            
            print(f"   📊 Координаты:")
            print(f"      • Везикулы - Y: {np.min(vesicle_coords[0])}-{np.max(vesicle_coords[0])}, X: {np.min(vesicle_coords[1])}-{np.max(vesicle_coords[1])}")
            print(f"      • Макрофаги - Y: {np.min(macrophage_coords[0])}-{np.max(macrophage_coords[0])}, X: {np.min(macrophage_coords[1])}-{np.max(macrophage_coords[1])}")
            
        except Exception as e:
            print(f"   ❌ Ошибка диагностики: {e}")