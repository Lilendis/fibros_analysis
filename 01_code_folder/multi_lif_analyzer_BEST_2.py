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
    def __init__(self, min_vesicle_size=5, max_vesicle_size=50, background_method='conservative', 
                subtraction_factor=0.7, vesicles_contrast_factor=1.0, protein_contrast_factor=1.5):
        self.min_vesicle_size = min_vesicle_size
        self.max_vesicle_size = max_vesicle_size
        self.background_method = background_method
        self.subtraction_factor = subtraction_factor
        self.vesicles_contrast_factor = vesicles_contrast_factor  # ⭐ НОВЫЙ ПАРАМЕТР
        self.protein_contrast_factor = protein_contrast_factor    # ⭐ НОВЫЙ ПАРАМЕТР
        self.igg_intensity_data = {}
        self.collagen_intensity_data = {}
        self.protein_intensities_from_igg = []
        print(f"      🎛️  Настройки анализатора: контраст везикул={vesicles_contrast_factor}, контраст белка={protein_contrast_factor}")
            
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
        """Анализ сохранения везикул после коррекции"""
        if original_vesicles is None or corrected_vesicles is None:
            return
        
        try:
            # Сегментация на оригинальном и скорректированном каналах
            original_binary = self.segment_vesicles(original_vesicles)
            corrected_binary = self.segment_vesicles(corrected_vesicles)
            
            if original_binary is not None and corrected_binary is not None:
                # Подсчет везикул через connected components
                original_labels = measure.label(original_binary)
                corrected_labels = measure.label(corrected_binary)
                
                original_count = np.max(original_labels) if np.max(original_labels) > 0 else 0
                corrected_count = np.max(corrected_labels) if np.max(corrected_labels) > 0 else 0
                
                preservation_rate = (corrected_count / original_count * 100) if original_count > 0 else 0
                
                print(f"   📊 СОХРАНЕНИЕ ВЕЗИКУЛ:")
                print(f"      • До коррекции: {original_count}")
                print(f"      • После коррекции: {corrected_count}")
                print(f"      • Сохранено: {preservation_rate:.1f}%")
                
                if preservation_rate < 70:  # Если потеряли более 30% везикул
                    print(f"      🚨 ВНИМАНИЕ: Потеряно более 30% везикул!")
                    return False
                elif preservation_rate > 95:  # Если сохранили почти все
                    print(f"      ✅ Отличное сохранение везикул!")
                    return True
                else:
                    print(f"      ⚠️  Умеренная потеря везикул")
                    return True
                    
        except Exception as e:
            print(f"   ❌ Ошибка анализа сохранения везикул: {e}")
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
        
        # Преобразуем в float для расчетов
        channel_float = channel_data.astype(np.float32)
        
        # Нормализуем к диапазону 0-1
        channel_normalized = channel_float / 255.0
        
        # Применяем gamma коррекцию для увеличения контраста
        gamma = 1.0 / contrast_factor
        channel_enhanced = np.power(channel_normalized, gamma)
        
        # Возвращаем к диапазону 0-255
        channel_enhanced = (channel_enhanced * 255).astype(np.uint8)
        
        return channel_enhanced
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
                threshold = np.percentile(channel_data, 85)  # Снизили с 95% до 85%
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
    
    def load_igg_from_file(self, igg_lif_file_path, output_base_dir):
        """Загружает ВСЕ образцы из IgG файла как контроли и сохраняет их изображения"""
        print(f"📖 Загрузка ВСЕХ образцов из IgG файла: {igg_lif_file_path}")
        
        if not os.path.exists(igg_lif_file_path):
            print(f"❌ IgG файл не найден: {igg_lif_file_path}")
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
            channel_thresholds = {} 
            self.protein_intensities_from_igg = []  # ⭐ СОХРАНЯЕМ ПОЛНЫЙ СПИСОК
            
            for i, image_item in enumerate(lif_file.image_list):
                if isinstance(image_item, dict):
                    image_name = image_item.get('name', f'image_{i+1}')
                else:
                    image_name = getattr(image_item, 'name', f'image_{i+1}')
                
                print(f"   🔍 Загрузка образца {i+1}/{total_images}: {image_name}")
                
                sample_channels = self.force_load_all_channels(image_item, igg_lif_file_path)
                
                if sample_channels:
                    igg_samples_found += 1
                    print(f"   ✅ Образец загружен: {image_name}")
                    
                    # Сохраняем IgG образцы так же, как основные образцы
                    self.save_igg_sample_images(sample_channels, image_name, igg_output_dir)
                    
                    for channel_idx, channel_data in sample_channels.items():
                        # Определяем пороговую интенсивность через анализ гистограммы
                        threshold = self.find_background_threshold(channel_data, min_pixel_count=5)
                        
                        if channel_idx not in channel_thresholds:
                            channel_thresholds[channel_idx] = []
                        
                        channel_thresholds[channel_idx].append(threshold)
                        print(f"      Канал {channel_idx}: порог исчезновения = {threshold:.2f}")
                        
                        # ОСОБАЯ ЛОГИКА ДЛЯ КАНАЛА БЕЛКА (4)
                        if channel_idx == 3:  # Канал 4 (белок)
                            max_intensity = np.max(channel_data)
                            self.protein_intensities_from_igg.append(max_intensity)  # ⭐ СОХРАНЯЕМ
                            print(f"      🎯 Канал белка (4): максимальная интенсивность = {max_intensity:.2f}")
                
            # Рассчитываем средние пороги для каждого канала
            for channel_idx, thresholds in channel_thresholds.items():
                self.igg_intensity_data[channel_idx] = np.mean(thresholds)
                print(f"📊 Канал {channel_idx}: средний порог исчезновения IgG = {self.igg_intensity_data[channel_idx]:.2f} "
                    f"(на основе {len(thresholds)} образцов)")
            
            # РАСЧЕТ IgG ФОНА ДЛЯ БЕЛКА
            if self.protein_intensities_from_igg:
                self.protein_igg_background = np.percentile(self.protein_intensities_from_igg, 90)  # 90% перцентиль
                print(f"🎯 IgG фон для белка (канал 4): {self.protein_igg_background:.2f}")
                print(f"   (рассчитан на основе {len(self.protein_intensities_from_igg)} образцов IgG)")
            else:
                self.protein_igg_background = 0
                print("⚠️  Не удалось определить IgG фон для белка")
            
            if igg_samples_found > 0:
                print(f"✅ УСПЕХ: Загружены IgG пороги из {igg_samples_found} образцов")
                print(f"📁 IgG изображения сохранены в: {igg_output_dir}")
                return True
            else:
                print("❌ ОШИБКА: Не удалось загрузить ни одного образца из IgG файла")
                return False
                
        except Exception as e:
            print(f"❌ Ошибка загрузки IgG файла: {e}")
            return False
    
    def find_igg_samples_in_main_file(self, samples_data):
        """Находит IgG образцы в основном файле и добавляет их в IgG данные"""
        print("🔍 Поиск IgG образцов в основном файле...")
        
        protein_intensities_from_main = []
        
        for mouse_id, samples in samples_data.items():
            for sample in samples:
                if self.detect_igg_samples(sample['sample_name']):
                    print(f"   ✅ Найден IgG образец: {sample['sample_name']}")
                    channels = sample['channels']
                    
                    if 3 in channels:  # Канал белка (4)
                        protein_channel = channels[3]
                        max_intensity = np.max(protein_channel)
                        protein_intensities_from_main.append(max_intensity)
                        print(f"      🎯 Канал белка (4): максимальная интенсивность = {max_intensity:.2f}")
        
        # Обновляем IgG фон для белка с учетом образцов из основного файла
        if protein_intensities_from_main:
            if hasattr(self, 'protein_intensities_from_igg') and self.protein_intensities_from_igg:
                # ⭐ ПРАВИЛЬНОЕ ОБЪЕДИНЕНИЕ: все интенсивности из обоих источников
                all_intensities = self.protein_intensities_from_igg + protein_intensities_from_main
                self.protein_igg_background = np.percentile(all_intensities, 90)
                print(f"🔄 Обновленный IgG фон для белка: {self.protein_igg_background:.2f}")
                print(f"   (объединены {len(self.protein_intensities_from_igg)} из IgG файла + {len(protein_intensities_from_main)} из основного файла)")
            else:
                # Если нет данных из IgG файла, используем только из основного файла
                self.protein_igg_background = np.percentile(protein_intensities_from_main, 90)
                self.protein_intensities_from_igg = []  # Инициализируем пустой список
                print(f"🎯 IgG фон для белка из основного файла: {self.protein_igg_background:.2f}")
        
        return len(protein_intensities_from_main) > 0
    
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
            gamma_values = {
                0: 0.8,   # Канал 0: Синий - Ядра
                2: 0.7,   # Канал 2: Красный - Везикулы
                3: 0.8    # Канал 3: Зеленый - Белок
            }
            gamma = gamma_values.get(channel_type, 0.8)
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
        
        # ⭐ ДИАГНОСТИКА: анализируем распределение интенсивностей
        if keep_background_black:
            print(f"      🔍 ДИАГНОСТИКА КАНАЛА:")
            print(f"         Min intensity: {np.min(gray_channel)}")
            print(f"         Max intensity: {np.max(gray_channel)}")
            print(f"         Mean intensity: {np.mean(gray_channel):.1f}")
            print(f"         Pixels > 0: {np.sum(gray_channel > 0)} / {gray_channel.size}")
            print(f"         Pixels > 10: {np.sum(gray_channel > 10)} / {gray_channel.size}")
            print(f"         Pixels > 50: {np.sum(gray_channel > 50)} / {gray_channel.size}")
            
            # ⭐ АВТОМАТИЧЕСКИЙ ПОРОГ на основе гистограммы
            hist, bins = np.histogram(gray_channel.flatten(), bins=256, range=[0,256])
            
            # Находим точку, где начинаются "настоящие" сигналы (пропускаем шум)
            cumulative = np.cumsum(hist)
            total_pixels = gray_channel.size
            threshold = 0
            
            for i in range(1, len(cumulative)):
                if cumulative[i] > total_pixels * 0.02:  # Первые 2% пикселей считаем шумом
                    threshold = i
                    break
            
            print(f"         Автоматический порог: {threshold}")
            
            # Используем автоматический порог + минимальное значение
            final_threshold = max(threshold, 15)  # Не менее 15
            mask = gray_channel > final_threshold
            
            print(f"         Используется порог: {final_threshold}")
            print(f"         Пикселей выше порога: {np.sum(mask)} / {gray_channel.size}")
            
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
            # Вычисляем рекомендованный процент вычитания
            collagen_90th = self.find_collagen_background_percentile(collagen_channel, 90)
            
            # Анализируем распределение интенсивностей везикул
            vesicles_max = np.max(vesicles_channel)
            vesicles_mean = np.mean(vesicles_channel)
            vesicles_90th = np.percentile(vesicles_channel, 90)
            
            print(f"\n   🔍 АНАЛИЗ ВЕЗИКУЛ ДЛЯ: {sample_name}")
            print(f"      📊 Коллаген (90%): {collagen_90th:.2f}")
            print(f"      📈 Везикулы - max: {vesicles_max:.2f}, mean: {vesicles_mean:.2f}, 90%: {vesicles_90th:.2f}")
            
            # Рекомендация процента вычитания
            if vesicles_max > 0:
                recommended_percent = min(80, (collagen_90th / vesicles_max) * 100)
                recommended_percent = max(20, recommended_percent)  # Не менее 20%
            else:
                recommended_percent = 50
            
            print(f"      💡 РЕКОМЕНДАЦИЯ: вычесть {recommended_percent:.1f}% от {collagen_90th:.2f}")
            
            # Интерактивный ввод
            while True:
                try:
                    user_input = '80'.strip()
                    
                    if user_input.lower() == 'skip':
                        print("      ⏭️  Пропущено вычитание фона везикул")
                        return vesicles_channel
                    elif user_input == '':
                        # Используем рекомендованное значение
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
            
            # Применяем вычитание
            amount_to_subtract = collagen_90th * subtraction_percent
            
            print(f"      🧮 Вычитание: {collagen_90th:.2f} × {subtraction_percent*100:.1f}% = {amount_to_subtract:.2f}")
            print(f"      📉 Везикулы до: max={vesicles_max:.2f}, mean={vesicles_mean:.2f}")
            
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
        """Создание композитного изображения с УЛУЧШЕННЫМИ коррекциями"""
        channels = sample_data['channels']
        sample_name = sample_data['sample_name']
        
        print(f"   🎨 Создание композита для: {sample_name}")
        print(f"   📊 Доступные каналы: {list(channels.keys())}")
        
        required_channels = [0, 2, 3]
        missing_channels = [ch for ch in required_channels if ch not in channels]
        if missing_channels:
            print(f"   ❌ Отсутствуют каналы: {missing_channels}")
            return None, None, None, None, None
        
        try:
            # Сохраняем СЫРЫЕ каналы
            protein_channel_raw = channels.get(3)
            vesicles_channel_raw = channels.get(2)  # Канал 2 - везикулы
            collagen_channel_raw = channels.get(1)  # Канал 1 - коллаген
            
            # ПРЕДВАРИТЕЛЬНАЯ КОРРЕКЦИЯ: интерактивное вычитание коллагена из везикул
            vesicles_channel_corrected = vesicles_channel_raw.copy()
            
            if collagen_channel_raw is not None and vesicles_channel_raw is not None:
                print("   🔧 ИНТЕРАКТИВНАЯ КОРРЕКЦИЯ ВЕЗИКУЛ...")
                vesicles_channel_corrected = self.subtract_vesicles_background_interactive(
                    vesicles_channel_raw, collagen_channel_raw, sample_name
                )
            
            # Предобработка каналов (используем УЖЕ СКОРРЕКТИРОВАННЫЕ везикулы)
            print("   🔧 Предобработка каналов...")
            nuclei_channel = self.preprocess_channel(channels[0], 0)
            vesicles_channel = self.preprocess_channel(vesicles_channel_corrected, 2)  # Скорректированные везикулы
            protein_channel = self.preprocess_channel(protein_channel_raw, 3)
            
            # ЗАДАЧА 1: Вычитание IgG фона из БЕЛКА (канал 4)
            protein_channel_before_correction = protein_channel.copy()  # Сохраняем ДО коррекции
            
            if hasattr(self, 'protein_igg_background') and self.protein_igg_background > 0:
                print("   🔧 Вычитание IgG фона из белка...")
                
                protein_before_max = np.max(protein_channel)
                protein_before_mean = np.mean(protein_channel)
                
                print(f"      IgG фон для белка: {self.protein_igg_background:.2f}")
                print(f"      Белок до коррекции: max={protein_before_max:.2f}, mean={protein_before_mean:.2f}")
                
                # Вычитаем 90% от IgG фона
                amount_to_subtract = self.protein_igg_background * 0.9
                
                # Защита от слишком агрессивного вычитания
                if amount_to_subtract >= protein_before_max * 0.8:
                    amount_to_subtract = protein_before_max * 0.5
                    print(f"      ⚠️  СЛИШКОМ МНОГО! Снижено до: {amount_to_subtract:.2f}")
                
                protein_channel_corrected = protein_channel.astype(np.float32) - amount_to_subtract
                protein_channel_corrected = np.clip(protein_channel_corrected, 0, 255).astype(np.uint8)
                
                protein_after_max = np.max(protein_channel_corrected)
                protein_after_mean = np.mean(protein_channel_corrected)
                
                print(f"      Вычтено: {amount_to_subtract:.2f} (90% от IgG фона)")
                print(f"      Белок после коррекции: max={protein_after_max:.2f}, mean={protein_after_mean:.2f}")
                
                protein_channel = protein_channel_corrected
            else:
                print("   ⚠️  IgG фон для белка не определен, коррекция не применена")
            
            # ⭐ УЛУЧШЕНИЕ ЯРКОСТИ ВЕЗИКУЛ - ТОЛЬКО ДЛЯ СКОРРЕКТИРОВАННЫХ ВЕЗИКУЛ
            print("   🔧 УЛУЧШЕНИЕ ЯРКОСТИ ВЕЗИКУЛ...")
            vesicles_channel_enhanced = self.enhance_vesicles_brightness(vesicles_channel)
            
            # ⭐ УВЕЛИЧЕНИЕ КОНТРАСТА - ТОЛЬКО ДЛЯ СКОРРЕКТИРОВАННЫХ КАНАЛОВ
            print("   🔧 Увеличение контраста СКОРРЕКТИРОВАННЫХ каналов...")
            print(f"      Везикулы: коэффициент {self.vesicles_contrast_factor}")
            print(f"      Белок: коэффициент {self.protein_contrast_factor}")
            
            # ⭐ КОНТРАСТ ПРИМЕНЯЕТСЯ ТОЛЬКО К СКОРРЕКТИРОВАННЫМ КАНАЛАМ
            vesicles_channel_final = self.enhance_contrast(vesicles_channel_enhanced, self.vesicles_contrast_factor)
            protein_channel_final = self.enhance_contrast(protein_channel, self.protein_contrast_factor)
            nuclei_channel_final = nuclei_channel  # Ядра без изменения контраста
            
            vesicles_channel = vesicles_channel_final
            protein_channel = protein_channel_final
            
            # Создание RGB композита
            print("   🖼️ Создание RGB композита...")
            height, width = nuclei_channel.shape
            composite = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Распределение каналов:
            composite[:,:,0] = nuclei_channel_final    # Синий - Ядра (без изменения контраста)
            composite[:,:,1] = protein_channel_final   # Зеленый - Белок (СКОРРЕКТИРОВАННЫЙ + контраст)  
            composite[:,:,2] = vesicles_channel_final  # Красный - Везикулы (СКОРРЕКТИРОВАННЫЕ + контраст)
            
            # Сегментация для визуализации (используем финальные каналы с контрастом)
            print("   🔍 Сегментация для визуализации...")
            vesicles_binary = self.segment_vesicles(vesicles_channel_final)
            macrophages_labels = self.segment_macrophages(nuclei_channel_final, protein_channel_final)
            
            # Сохранение изображений
            safe_name = "".join(c for c in sample_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            
            # 1. Сохраняем композит БЕЗ аннотаций
            composite_filename = f"{safe_name}_composite.png"
            composite_path = os.path.join(output_dir, composite_filename)
            
            # 2. Сохраняем композит С аннотациями
            composite_annotated = self.draw_vesicles_and_macrophages_on_composite(
                composite, vesicles_binary, macrophages_labels,
                vesicles_color=(0, 255, 255), macrophages_color=(255, 255, 255), thickness=2
            )
            composite_annotated_filename = f"{safe_name}_composite_annotated.png"
            composite_annotated_path = os.path.join(output_dir, composite_annotated_filename)
            
            # 3. Сохраняем ИСХОДНЫЙ канал везикул (без коррекции и без контраста)
            vesicles_original_colored = self.apply_colored_channel(vesicles_channel_raw, 'red', keep_background_black=True)
            vesicles_original_colored = self.add_scale_bar(vesicles_original_colored, scale_length_pixels=100, scale_text="100 μm")
            vesicles_original_filename = f"{safe_name}_channel2_vesicles_ORIGINAL.png"
            vesicles_original_path = os.path.join(output_dir, vesicles_original_filename)
            cv2.imwrite(vesicles_original_path, vesicles_original_colored)
            print(f"   ✅ Исходный канал везикул сохранен: {vesicles_original_filename}")
            
            # 4. Сохраняем ИСХОДНЫЙ канал белка (без коррекции и без контраста)
            protein_original_colored = self.apply_colored_channel(protein_channel_before_correction, 'green', keep_background_black=True)
            protein_original_colored = self.add_scale_bar(protein_original_colored, scale_length_pixels=100, scale_text="100 μm")
            protein_original_filename = f"{safe_name}_channel3_protein_ORIGINAL.png"
            protein_original_path = os.path.join(output_dir, protein_original_filename)
            cv2.imwrite(protein_original_path, protein_original_colored)
            print(f"   ✅ Исходный канал белка сохранен: {protein_original_filename}")
            
            # 5. Сохраняем канал везикул С кругами (СКОРРЕКТИРОВАННЫЙ + контраст)
            vesicles_colored = self.apply_colored_channel(vesicles_channel_final, 'red', keep_background_black=True)
            vesicles_colored = self.apply_black_background(vesicles_colored, vesicles_channel_final, threshold=10)
            vesicles_annotated = self.draw_vesicles_on_channel(
                vesicles_colored, vesicles_binary, color=(0, 255, 0), thickness=2
            )
            vesicles_annotated_filename = f"{safe_name}_channel2_vesicles_annotated.png"
            vesicles_annotated_path = os.path.join(output_dir, vesicles_annotated_filename)
            
            # 6. Сохраняем КОРРЕКТИРОВАННЫЙ канал везикул (без кругов, С контрастом)
            vesicles_corrected_colored = self.apply_colored_channel(vesicles_channel_final, 'red', keep_background_black=True)
            vesicles_corrected_colored = self.add_scale_bar(vesicles_corrected_colored, scale_length_pixels=100, scale_text="100 μm")
            vesicles_corrected_filename = f"{safe_name}_channel2_vesicles_CORRECTED.png"
            vesicles_corrected_path = os.path.join(output_dir, vesicles_corrected_filename)
            cv2.imwrite(vesicles_corrected_path, vesicles_corrected_colored)
            print(f"   ✅ Скорректированный канал везикул сохранен: {vesicles_corrected_filename}")
            
            # 7. Сохраняем КОРРЕКТИРОВАННЫЙ канал белка (без кругов, С контрастом)
            protein_corrected_colored = self.apply_colored_channel(protein_channel_final, 'green', keep_background_black=True)
            protein_corrected_colored = self.add_scale_bar(protein_corrected_colored, scale_length_pixels=100, scale_text="100 μm")
            protein_corrected_filename = f"{safe_name}_channel3_protein_CORRECTED.png"
            protein_corrected_path = os.path.join(output_dir, protein_corrected_filename)
            cv2.imwrite(protein_corrected_path, protein_corrected_colored)
            print(f"   ✅ Скорректированный канал белка сохранен: {protein_corrected_filename}")
            
            # 8. Сохраняем остальные каналы
            nuclei_colored = self.apply_colored_channel(nuclei_channel_final, 'blue', keep_background_black=True)

            # ⭐ ПРИНУДИТЕЛЬНОЕ ОБНУЛЕНИЕ ФОНА ДЛЯ ВЕЗИКУЛ
            if vesicles_channel_final is not None:
                # Находим порог для фона
                vesicles_threshold = np.percentile(vesicles_channel_final, 85)  # 85% перцентиль
                background_mask = vesicles_channel_final < vesicles_threshold
                
                # Обнуляем фон в композитном изображении
                composite[background_mask, 2] = 0  # Красный канал
                
                # Обнуляем фон в отдельных каналах
                if 'vesicles_colored' in locals():
                    vesicles_colored[background_mask] = [0, 0, 0]
                if 'vesicles_corrected_colored' in locals():
                    vesicles_corrected_colored[background_mask] = [0, 0, 0]
                
                print(f"   🎯 Принудительное обнуление фона везикул (порог: {vesicles_threshold:.1f})")
            
            # Добавляем шкалу ко всем изображениям
            composite_with_scale = self.add_scale_bar(composite, scale_length_pixels=100, scale_text="100 μm")
            composite_annotated_with_scale = self.add_scale_bar(composite_annotated, scale_length_pixels=100, scale_text="100 μm")
            vesicles_annotated_with_scale = self.add_scale_bar(vesicles_annotated, scale_length_pixels=100, scale_text="100 μm")
            nuclei_colored_with_scale = self.add_scale_bar(nuclei_colored, scale_length_pixels=100, scale_text="100 μm")
            
            # Сохраняем изображения со шкалой
            cv2.imwrite(composite_path, composite_with_scale)
            cv2.imwrite(composite_annotated_path, composite_annotated_with_scale)
            cv2.imwrite(vesicles_annotated_path, vesicles_annotated_with_scale)
            cv2.imwrite(os.path.join(output_dir, f"{safe_name}_channel0_nuclei.png"), nuclei_colored_with_scale)
            
            # Сохраняем коллаген если есть
            if 1 in channels:
                collagen_channel_processed = self.preprocess_channel(channels[1], 1)
                if collagen_channel_processed is not None:
                    collagen_colored = self.apply_colored_channel(collagen_channel_processed, 'green', keep_background_black=True)
                    collagen_colored = self.add_scale_bar(collagen_colored, scale_length_pixels=100, scale_text="100 μm")
                    cv2.imwrite(os.path.join(output_dir, f"{safe_name}_channel1_collagen.png"), collagen_colored)
            
            print(f"   ✅ Все изображения сохранены для: {sample_name}")
            print(f"   📁 Доступные файлы:")
            print(f"      • {composite_filename} - композит (все каналы)")
            print(f"      • {composite_annotated_filename} - композит с аннотациями") 
            print(f"      • {vesicles_original_filename} - исходные везикулы (без изменений)")
            print(f"      • {vesicles_corrected_filename} - скорректированные везикулы (с контрастом)")
            print(f"      • {vesicles_annotated_filename} - везикулы с кругами")
            print(f"      • {protein_original_filename} - исходный белок (без изменений)")
            print(f"      • {protein_corrected_filename} - скорректированный белок (с контрастом)")
            
            # ⭐ ВОЗВРАЩАЕМ ФИНАЛЬНЫЕ КАНАЛЫ С КОНТРАСТОМ ДЛЯ СЕГМЕНТАЦИИ
            return composite, nuclei_channel_final, vesicles_channel_final, protein_channel_final, composite_filename
            
        except Exception as e:
            print(f"   ❌ Ошибка создания композита: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None, None, None

    
    def segment_vesicles(self, vesicles_channel):
        """Сегментация везикул с улучшенной обработкой"""
        if vesicles_channel is None:
            return None
            
        try:
            print("   🔍 Сегментация везикул...")
            
            # Анализ интенсивности перед сегментацией
            channel_max = np.max(vesicles_channel)
            channel_mean = np.mean(vesicles_channel)
            print(f"      Интенсивность: max={channel_max:.1f}, mean={channel_mean:.1f}")
            
            # Если изображение слишком темное, предупреждаем
            if channel_max < 50:
                print("      ⚠️  ВНИМАНИЕ: Очень темное изображение, сегментация может быть неточной")
            
            # Легкое размытие для уменьшения шума (меньше размытия для сохранения деталей)
            blurred = cv2.GaussianBlur(vesicles_channel, (1, 1), 0)
            
            # Адаптивный порог с более чувствительными настройками
            binary = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 6  # Более чувствительные параметры
            )
            
            # Морфологические операции для очистки
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))  # Меньшее ядро
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

            
            # Удаление мелких объектов
            binary_bool = binary.astype(bool)
            cleaned = morphology.remove_small_objects(binary_bool, min_size=self.min_vesicle_size)
            
            # Метка компонентов
            labeled = measure.label(cleaned)
            regions = measure.regionprops(labeled)
            
            # Создание маски с менее строгими критериями
            mask = np.zeros_like(cleaned, dtype=bool)
            vesicle_count = 0
            
            for region in regions:
                size_ok = self.min_vesicle_size <= region.area <= self.max_vesicle_size
                shape_ok = region.eccentricity < 0.97  # Более либеральный порог
                
                if size_ok and shape_ok:
                    coords = region.coords
                    mask[coords[:, 0], coords[:, 1]] = True
                    vesicle_count += 1
            
            result = mask.astype(np.uint8) * 255
            
            print(f"   ✅ Сегментировано везикул: {vesicle_count}")
            
            # Диагностика
            if vesicle_count == 0:
                print("   🚨 ПРЕДУПРЕЖДЕНИЕ: Не найдено везикул!")
                print("   💡 Возможные причины:")
                print("      • Слишком агрессивное вычитание фона")
                print("      • Слишком темное изображение")
                print("      • Слишком строгие параметры сегментации")
            
            return result
                
        except Exception as e:
            print(f"   ❌ Ошибка сегментации везикул: {e}")
            return None
    
    def segment_macrophages(self, nuclei_channel, protein_channel):
        """Сегментация макрофагов"""
        if nuclei_channel is None or protein_channel is None:
            return None
            
        try:
            print("   🔍 Сегментация макрофагов...")
            
            nuclei_threshold = filters.threshold_otsu(nuclei_channel)
            nuclei_binary = nuclei_channel > nuclei_threshold
            
            protein_threshold = filters.threshold_otsu(protein_channel)
            protein_binary = protein_channel > protein_threshold
            
            macrophages_binary = np.logical_and(nuclei_binary, protein_binary)
            macrophages_binary = ndimage.binary_fill_holes(macrophages_binary)
            macrophages_binary = morphology.remove_small_objects(macrophages_binary, min_size=50)
            
            distance = ndimage.distance_transform_edt(macrophages_binary)
            coordinates = peak_local_max(distance, min_distance=10, labels=macrophages_binary)
            
            local_maxi = np.zeros_like(distance, dtype=bool)
            local_maxi[tuple(coordinates.T)] = True
            
            markers = measure.label(local_maxi)
            labels = segmentation.watershed(-distance, markers, mask=macrophages_binary)
            
            macrophage_count = np.max(labels) if np.max(labels) > 0 else 0
            print(f"   ✅ Сегментировано макрофагов: {macrophage_count}")
            
            return labels
            
        except Exception as e:
            print(f"   ❌ Ошибка сегментации макрофагов: {e}")
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
                    'macrophages_count': len(overlapping_macrophages)
                })
            
            percentage = (colocalized_vesicles / total_vesicles * 100) if total_vesicles > 0 else 0
            
            # ⭐ АНАЛИЗ ВЕЗИКУЛ В КЛЕТКАХ (пересечение с ядрами)
            vesicles_in_cells = self.analyze_vesicles_in_cells(vesicles_binary, nuclei_channel, sample_name)
            
            print(f"   ✅ Везикул в макрофагах: {colocalized_vesicles} ({percentage:.1f}%)")
            
            return percentage, colocalized_vesicles, total_vesicles, vesicles_in_cells, colocalization_data  # ⭐ ВОЗВРАЩАЕМ 5 ЗНАЧЕНИЙ
            
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
        
        composite, nuclei_processed, vesicles_processed, protein_processed, composite_filename = result
        
        print(f"   🔍 Анализ колокализации...")
        vesicles_binary = self.segment_vesicles(vesicles_processed)
        macrophages_labels = self.segment_macrophages(nuclei_processed, protein_processed)
        
        if vesicles_binary is not None and macrophages_labels is not None:
            # ⭐ ОБНОВЛЕННЫЙ ВЫЗОВ - добавлен nuclei_processed и получаем 5 значений
            percentage, colocalized, total, vesicles_in_cells, colocalization_data = self.analyze_colocalization(
                vesicles_binary, macrophages_labels, nuclei_processed, sample_name
            )
            
            safe_name = "".join(c for c in sample_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            csv_filename = f"{safe_name}_results.csv"
            csv_path = os.path.join(mouse_dir, csv_filename)
            
            results_df = pd.DataFrame(colocalization_data)
            results_df.to_csv(csv_path, index=False, encoding='utf-8')
            
            # ⭐ ОБНОВЛЕННЫЙ SUMMARY - добавлено vesicles_in_cells
            summary = {
                'sample_name': sample_name,
                'mouse_id': mouse_id,
                'composite_image': composite_filename,
                'results_csv': csv_filename,
                'total_vesicles': total,
                'colocalized_vesicles': colocalized,
                'vesicles_in_cells': vesicles_in_cells,  # ⭐ НОВОЕ ПОЛЕ
                'colocalization_percentage': percentage,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            print(f"   ✅ РЕЗУЛЬТАТЫ:")
            print(f"      • Везикул всего: {total}")
            print(f"      • Везикул в макрофагах: {colocalized}") 
            print(f"      • Везикул в клетках: {vesicles_in_cells}")  # ⭐ НОВАЯ СТАТИСТИКА
            print(f"      • Процент колокализации: {percentage:.2f}%")
            
            # ⭐ ДОПОЛНИТЕЛЬНАЯ СТАТИСТИКА
            if total > 0:
                in_cells_percentage = (vesicles_in_cells / total) * 100
                print(f"      • Процент везикул в клетках: {in_cells_percentage:.2f}%")
            
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
        """Сильно увеличивает яркость везикул для лучшего обнаружения"""
        if vesicles_channel is None:
            return vesicles_channel
        
        try:
            # Анализируем текущее распределение интенсивностей
            current_max = np.max(vesicles_channel)
            current_95th = np.percentile(vesicles_channel, 95)
            current_mean = np.mean(vesicles_channel)
            current_median = np.median(vesicles_channel)
            
            print(f"      📊 Везикулы до улучшения: max={current_max:.1f}, 95%={current_95th:.1f}, mean={current_mean:.1f}, median={current_median:.1f}")
            
            # Если изображение слишком темное, применяем агрессивное усиление
            if current_max < 100:
                print(f"      ⚠️  Изображение темное, применяем агрессивное усиление (коэффициент: {enhancement_factor}x)")
                
                # Метод 1: Умножение интенсивностей
                vesicles_enhanced = vesicles_channel.astype(np.float32) * enhancement_factor
                vesicles_enhanced = np.clip(vesicles_enhanced, 0, 255).astype(np.uint8)
                
            else:
                # Метод 2: Контрастное растяжение для ярких изображений
                # Увеличиваем контраст только в верхнем диапазоне интенсивностей
                low_percentile = np.percentile(vesicles_channel, 40)
                high_percentile = np.percentile(vesicles_channel, 98)
                
                # Растягиваем гистограмму
                vesicles_enhanced = vesicles_channel.astype(np.float32)
                vesicles_enhanced = (vesicles_enhanced - low_percentile) * (255.0 / (high_percentile - low_percentile))
                vesicles_enhanced = np.clip(vesicles_enhanced, 0, 255).astype(np.uint8)
                
                # Дополнительное легкое усиление
                vesicles_enhanced = cv2.convertScaleAbs(vesicles_enhanced, alpha=1.3, beta=10)
            
            # Применяем CLAHE для локального контраста
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            vesicles_enhanced = clahe.apply(vesicles_enhanced)
            
            # Легкое размытие для уменьшения шума
            vesicles_enhanced = cv2.GaussianBlur(vesicles_enhanced, (3, 3), 0)
            
            after_max = np.max(vesicles_enhanced)
            after_95th = np.percentile(vesicles_enhanced, 95)
            after_mean = np.mean(vesicles_enhanced)
            
            print(f"      ✅ Везикулы после улучшения: max={after_max:.1f}, 95%={after_95th:.1f}, mean={after_mean:.1f}")
            print(f"      🔧 Коэффициент усиления: {enhancement_factor}x")
            
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
    