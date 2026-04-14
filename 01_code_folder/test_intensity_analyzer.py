"""
ТЕСТОВЫЙ СКРИПТ ДЛЯ ОПРЕДЕЛЕНИЯ АБСОЛЮТНОЙ ЯРКОСТИ ВЕЗИКУЛ
Сегментирует везикулы и показывает их интенсивность
"""

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from readlif.reader import LifFile
from skimage import filters, measure, morphology, segmentation
from scipy import ndimage
from datetime import datetime
import traceback
from PIL import Image, ImageDraw, ImageFont

class VesicleIntensityTester:
    """Тестер для определения оптимальной яркости везикул"""
    
    def __init__(self, lif_file_path, output_dir="vesicle_intensity_test"):
        """
        Args:
            lif_file_path: путь к LIF файлу
            output_dir: папка для сохранения результатов
        """
        self.lif_file_path = lif_file_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Параметры сегментации (можно настраивать)
        self.min_vesicle_size = 5
        self.max_vesicle_size = 200
        self.circularity_threshold = 0.6  # Минимальная "круглость"
        
        print("=" * 80)
        print("🔬 ТЕСТОВЫЙ АНАЛИЗАТОР ИНТЕНСИВНОСТИ ВЕЗИКУЛ")
        print("=" * 80)
        print(f"📁 Входной файл: {lif_file_path}")
        print(f"📁 Выходная папка: {output_dir}")
        print("=" * 80)
    
    def load_channel(self, image_item, channel_idx=2):
        """Загружает указанный канал из образца"""
        try:
            from PIL import Image
            
            # Получаем имя образца
            if isinstance(image_item, dict):
                image_name = image_item.get('name', 'unknown')
                num_channels = image_item.get('channels', 4)
            else:
                image_name = getattr(image_item, 'name', 'unknown')
                num_channels = 4
            
            print(f"\n📷 Загрузка: {image_name}")
            
            # Загружаем LIF файл
            lif_file = LifFile(self.lif_file_path)
            
            # Находим индекс изображения
            image_index = None
            for i, img_item in enumerate(lif_file.image_list):
                img_name = img_item.get('name', '') if isinstance(img_item, dict) else getattr(img_item, 'name', '')
                if img_name == image_name:
                    image_index = i
                    break
            
            if image_index is None:
                print(f"   ⚠️  Изображение не найдено")
                return None, image_name
            
            # Получаем объект изображения
            lif_image = lif_file.get_image(image_index)
            
            # Загружаем канал
            channel_data = lif_image.get_frame(c=channel_idx)
            
            if channel_data is not None:
                # Конвертируем в numpy array
                if isinstance(channel_data, Image.Image):
                    channel_array = np.array(channel_data)
                else:
                    channel_array = channel_data
                
                # Конвертируем в 8-bit если нужно
                if channel_array.dtype != np.uint8:
                    if channel_array.dtype == np.uint16:
                        channel_array = (channel_array / 256).astype(np.uint8)
                    else:
                        channel_array = cv2.normalize(
                            channel_array, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
                        )
                
                print(f"   ✅ Канал {channel_idx} загружен: {channel_array.shape}")
                return channel_array, image_name
            
            return None, image_name
            
        except Exception as e:
            print(f"   ❌ Ошибка загрузки: {e}")
            return None, "unknown"
    
    def segment_vesicles(self, image):
        """
        Сегментирует везикулы (круглые/овальные яркие объекты)
        
        Returns:
            vesicles_mask: бинарная маска везикул
            vesicle_stats: список словарей со статистикой по каждой везикуле
        """
        if image is None:
            return None, []
        
        try:
            print(f"   🔍 Сегментация везикул...")
            
            # 1. Предобработка для улучшения контраста
            # Медианный фильтр для удаления шума
            img_denoised = cv2.medianBlur(image, 3)
            
            # 2. Локальное усиление контраста (CLAHE)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img_enhanced = clahe.apply(img_denoised)
            
            # 3. Адаптивный порог Otsu
            try:
                threshold = filters.threshold_otsu(img_enhanced)
                binary = img_enhanced > threshold
            except:
                # Fallback: если Otsu не работает, используем фиксированный порог
                binary = img_enhanced > 20
            
            # 4. Морфологическая очистка
            # Убираем мелкие объекты (шум)
            binary = morphology.remove_small_objects(binary, min_size=self.min_vesicle_size)
            
            # Заполняем маленькие дыры
            binary = ndimage.binary_fill_holes(binary)
            
            # 5. Разделяем соединенные объекты с помощью watershed
            # Вычисляем расстояние до фона
            distance = ndimage.distance_transform_edt(binary)
            
            # Находим локальные максимумы (центры везикул)
            from skimage.feature import peak_local_max
            coordinates = peak_local_max(
                distance, 
                min_distance=5,
                exclude_border=False,
                num_peaks=500
            )
            
            if len(coordinates) == 0:
                # Если нет локальных максимумов, используем простую маркировку
                markers = measure.label(binary)
            else:
                # Создаем маркеры для watershed
                markers = np.zeros_like(distance, dtype=np.int32)
                for i, (y, x) in enumerate(coordinates):
                    if binary[y, x]:
                        markers[y, x] = i + 1
                
                # Watershed сегментация
                labels = segmentation.watershed(-distance, markers, mask=binary)
            
            # 6. Анализ каждой везикулы
            vesicle_stats = []
            vesicles_mask = np.zeros_like(image, dtype=np.uint8)
            
            regions = measure.regionprops(labels, intensity_image=image)
            
            for i, region in enumerate(regions):
                # Основные параметры
                area = region.area
                
                # Пропускаем слишком маленькие
                if area < self.min_vesicle_size:
                    continue
                
                # Пропускаем слишком большие (возможно скопления)
                if area > self.max_vesicle_size * 2:
                    continue
                
                # Вычисляем круглость (1 = идеальный круг)
                perimeter = region.perimeter
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                else:
                    circularity = 0
                
                # Отношение сторон (1 = квадрат/круг)
                y0, x0, y1, x1 = region.bbox
                width = x1 - x0
                height = y1 - y0
                aspect_ratio = max(width, height) / min(width, height) if min(width, height) > 0 else 1
                
                # Интенсивность
                mean_intensity = region.mean_intensity
                max_intensity = np.max(image[labels == region.label])
                
                # Сохраняем если достаточно круглое
                if circularity > self.circularity_threshold and aspect_ratio < 2.5:
                    mask = labels == region.label
                    vesicles_mask[mask] = 255
                    
                    vesicle_stats.append({
                        'id': len(vesicle_stats) + 1,
                        'area': area,
                        'circularity': circularity,
                        'aspect_ratio': aspect_ratio,
                        'mean_intensity': mean_intensity,
                        'max_intensity': max_intensity,
                        'centroid_y': region.centroid[0],
                        'centroid_x': region.centroid[1],
                        'bbox': region.bbox
                    })
            
            print(f"   ✅ Найдено везикул: {len(vesicle_stats)}")
            return vesicles_mask, vesicle_stats
            
        except Exception as e:
            print(f"   ❌ Ошибка сегментации: {e}")
            traceback.print_exc()
            return None, []
    
    def create_visualization(self, original_image, vesicles_mask, vesicle_stats, sample_name):
        """
        Создает визуализацию с красными везикулами
        """
        try:
            # Создаем RGB изображение
            height, width = original_image.shape
            vis_image = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Красный канал = оригинальная интенсивность
            vis_image[:,:,2] = original_image
            
            # Рисуем контуры и подписи
            if vesicles_mask is not None and np.any(vesicles_mask):
                # Находим контуры везикул
                contours, _ = cv2.findContours(vesicles_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Рисуем контуры зеленым (для контраста с красным)
                cv2.drawContours(vis_image, contours, -1, (0, 255, 0), 1)
                
                # Добавляем подписи
                for stats in vesicle_stats:
                    cx = int(stats['centroid_x'])
                    cy = int(stats['centroid_y'])
                    
                    # Желтый центр
                    cv2.circle(vis_image, (cx, cy), 3, (0, 255, 255), -1)
                    
                    # Белая подпись
                    label = f"#{stats['id']}: {stats['mean_intensity']:.0f}"
                    cv2.putText(vis_image, label, (cx + 5, cy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            return vis_image
            
        except Exception as e:
            print(f"   ❌ Ошибка визуализации: {e}")
            return original_image
    
    def analyze_intensity_distribution(self, vesicle_stats):
        """
        Анализирует распределение интенсивностей везикул
        """
        if not vesicle_stats:
            return None
        
        intensities = [v['mean_intensity'] for v in vesicle_stats]
        max_intensities = [v['max_intensity'] for v in vesicle_stats]
        
        stats = {
            'count': len(intensities),
            'mean': np.mean(intensities),
            'median': np.median(intensities),
            'std': np.std(intensities),
            'min': np.min(intensities),
            'max': np.max(intensities),
            'q25': np.percentile(intensities, 25),
            'q75': np.percentile(intensities, 75),
            'q90': np.percentile(intensities, 90),
            'max_mean': np.mean(max_intensities),
            'max_median': np.median(max_intensities)
        }
        
        return stats
    
    def plot_results(self, original_image, vesicle_stats, sample_name):
        """
        Создает графики для анализа
        """
        try:
            if not vesicle_stats:
                return
            
            intensities = [v['mean_intensity'] for v in vesicle_stats]
            areas = [v['area'] for v in vesicle_stats]
            circularities = [v['circularity'] for v in vesicle_stats]
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # 1. Гистограмма интенсивностей
            axes[0, 0].hist(intensities, bins=20, color='red', alpha=0.7, edgecolor='black')
            axes[0, 0].axvline(np.mean(intensities), color='blue', linestyle='--', label=f'Mean: {np.mean(intensities):.1f}')
            axes[0, 0].axvline(np.median(intensities), color='green', linestyle='--', label=f'Median: {np.median(intensities):.1f}')
            axes[0, 0].set_xlabel('Средняя интенсивность')
            axes[0, 0].set_ylabel('Количество везикул')
            axes[0, 0].set_title('Распределение интенсивностей')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Зависимость интенсивности от площади
            axes[0, 1].scatter(areas, intensities, alpha=0.6, color='purple')
            axes[0, 1].set_xlabel('Площадь (пиксели)')
            axes[0, 1].set_ylabel('Интенсивность')
            axes[0, 1].set_title('Интенсивность vs Площадь')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Гистограмма площадей
            axes[0, 2].hist(areas, bins=20, color='blue', alpha=0.7, edgecolor='black')
            axes[0, 2].set_xlabel('Площадь (пиксели)')
            axes[0, 2].set_ylabel('Количество')
            axes[0, 2].set_title('Распределение размеров')
            axes[0, 2].grid(True, alpha=0.3)
            
            # 4. Круглость везикул
            axes[1, 0].hist(circularities, bins=20, color='green', alpha=0.7, edgecolor='black')
            axes[1, 0].axvline(self.circularity_threshold, color='red', linestyle='--', label=f'Порог: {self.circularity_threshold}')
            axes[1, 0].set_xlabel('Круглость (1 = идеальный круг)')
            axes[1, 0].set_ylabel('Количество')
            axes[1, 0].set_title('Распределение круглости')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # 5. Box plot интенсивностей
            axes[1, 1].boxplot(intensities)
            axes[1, 1].set_ylabel('Интенсивность')
            axes[1, 1].set_title('Box plot интенсивностей')
            axes[1, 1].grid(True, alpha=0.3)
            
            # 6. Текстовая статистика
            stats_text = f"""
            Всего везикул: {len(intensities)}
            
            Интенсивность:
            • Средняя: {np.mean(intensities):.1f}
            • Медиана: {np.median(intensities):.1f}
            • 25%: {np.percentile(intensities, 25):.1f}
            • 75%: {np.percentile(intensities, 75):.1f}
            • 90%: {np.percentile(intensities, 90):.1f}
            • Min: {np.min(intensities):.1f}
            • Max: {np.max(intensities):.1f}
            
            Площадь:
            • Средняя: {np.mean(areas):.1f}
            • Медиана: {np.median(areas):.1f}
            
            Рекомендуемый порог: {np.percentile(intensities, 25):.0f}-{np.percentile(intensities, 10):.0f}
            """
            
            axes[1, 2].text(0.1, 0.5, stats_text, fontsize=10, 
                          verticalalignment='center', fontfamily='monospace',
                          transform=axes[1, 2].transAxes)
            axes[1, 2].axis('off')
            
            plt.suptitle(f'Анализ везикул: {sample_name}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # Сохраняем
            safe_name = "".join(c for c in sample_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            plot_path = os.path.join(self.output_dir, f"{safe_name}_analysis.png")
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"   📊 Графики сохранены: {plot_path}")
            
        except Exception as e:
            print(f"   ⚠️ Ошибка создания графиков: {e}")
    
    def save_results(self, original_image, vesicles_mask, vis_image, vesicle_stats, sample_name):
        """
        Сохраняет все результаты с красным цветом для везикул
        """
        try:
            safe_name = "".join(c for c in sample_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            
            # 1. Сохраняем оригинальное изображение (серое)
            orig_path = os.path.join(self.output_dir, f"{safe_name}_original.png")
            cv2.imwrite(orig_path, original_image)
            print(f"   ✅ Оригинал (серый): {safe_name}_original.png")
            
            # 2. Сохраняем КРАСНОЕ изображение везикул
            # Создаем RGB изображение с красным каналом
            height, width = original_image.shape
            red_image = np.zeros((height, width, 3), dtype=np.uint8)
            red_image[:,:,2] = original_image  # Красный канал (BGR: 0=Blue, 1=Green, 2=Red)
            
            red_path = os.path.join(self.output_dir, f"{safe_name}_red.png")
            cv2.imwrite(red_path, red_image)
            print(f"   🔴 Красный канал: {safe_name}_red.png")
            
            # 3. Сохраняем маску везикул
            if vesicles_mask is not None:
                mask_path = os.path.join(self.output_dir, f"{safe_name}_mask.png")
                cv2.imwrite(mask_path, vesicles_mask)
                print(f"   ⚪ Маска: {safe_name}_mask.png")
            
            # 4. Сохраняем визуализацию (уже цветная из метода create_visualization)
            vis_path = os.path.join(self.output_dir, f"{safe_name}_annotated.png")
            cv2.imwrite(vis_path, vis_image)
            print(f"   🎯 Аннотированное: {safe_name}_annotated.png")
            
            # 5. Сохраняем КРАСНОЕ изображение с аннотациями
            if vesicles_mask is not None:
                # Создаем красную версию с аннотациями
                red_annotated = red_image.copy()
                
                # Находим контуры везикул
                contours, _ = cv2.findContours(vesicles_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Рисуем контуры зеленым
                cv2.drawContours(red_annotated, contours, -1, (0, 255, 0), 1)
                
                # Добавляем подписи с интенсивностью
                for stats in vesicle_stats:
                    cx = int(stats['centroid_x'])
                    cy = int(stats['centroid_y'])
                    
                    # Рисуем центр
                    cv2.circle(red_annotated, (cx, cy), 3, (0, 255, 255), -1)
                    
                    # Подпись
                    label = f"#{stats['id']}: {stats['mean_intensity']:.0f}"
                    cv2.putText(red_annotated, label, (cx + 5, cy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                red_annotated_path = os.path.join(self.output_dir, f"{safe_name}_red_annotated.png")
                cv2.imwrite(red_annotated_path, red_annotated)
                print(f"   🔴🎯 Красный с аннотациями: {safe_name}_red_annotated.png")
            
            # 6. Сохраняем статистику в CSV
            if vesicle_stats:
                import csv
                csv_path = os.path.join(self.output_dir, f"{safe_name}_stats.csv")
                
                with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=['id', 'area', 'circularity', 'aspect_ratio', 
                                                        'mean_intensity', 'max_intensity',
                                                        'centroid_y', 'centroid_x', 'bbox'])
                    writer.writeheader()
                    writer.writerows(vesicle_stats)
                
                print(f"   📊 Статистика: {safe_name}_stats.csv")
            
            # 7. Сохраняем текстовый отчет
            if vesicle_stats:
                txt_path = os.path.join(self.output_dir, f"{safe_name}_report.txt")
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(f"ОТЧЕТ ПО ВЕЗИКУЛАМ: {sample_name}\n")
                    f.write("=" * 60 + "\n\n")
                    
                    intensities = [v['mean_intensity'] for v in vesicle_stats]
                    areas = [v['area'] for v in vesicle_stats]
                    
                    f.write(f"Всего везикул: {len(vesicle_stats)}\n\n")
                    
                    f.write("ИНТЕНСИВНОСТЬ:\n")
                    f.write(f"  Средняя: {np.mean(intensities):.1f}\n")
                    f.write(f"  Медиана: {np.median(intensities):.1f}\n")
                    f.write(f"  Std: {np.std(intensities):.1f}\n")
                    f.write(f"  Min: {np.min(intensities):.1f}\n")
                    f.write(f"  Max: {np.max(intensities):.1f}\n")
                    f.write(f"  25%: {np.percentile(intensities, 25):.1f}\n")
                    f.write(f"  50%: {np.percentile(intensities, 50):.1f}\n")
                    f.write(f"  75%: {np.percentile(intensities, 75):.1f}\n")
                    f.write(f"  90%: {np.percentile(intensities, 90):.1f}\n\n")
                    
                    f.write("ПЛОЩАДЬ:\n")
                    f.write(f"  Средняя: {np.mean(areas):.1f}\n")
                    f.write(f"  Медиана: {np.median(areas):.1f}\n")
                    f.write(f"  Min: {np.min(areas):.1f}\n")
                    f.write(f"  Max: {np.max(areas):.1f}\n\n")
                    
                    f.write("РЕКОМЕНДАЦИИ:\n")
                    f.write(f"  Порог для отсечения фона: {np.percentile(intensities, 25):.0f} (25% перцентиль)\n")
                    f.write(f"  Консервативный порог: {np.percentile(intensities, 10):.0f} (10% перцентиль)\n")
                    f.write(f"  Агрессивный порог: {np.percentile(intensities, 50):.0f} (медиана)\n")
                
                print(f"   📄 Отчет: {safe_name}_report.txt")
            
            print(f"   ✅ Все файлы сохранены в: {self.output_dir}")
            
        except Exception as e:
            print(f"   ❌ Ошибка сохранения: {e}")
            traceback.print_exc()
    
    def process_samples(self, num_samples=3, start_from=0):
        """
        Обрабатывает несколько образцов
        
        Args:
            num_samples: количество образцов для анализа
            start_from: начальный индекс (для пропуска первых)
        """
        try:
            # Загружаем LIF файл
            lif_file = LifFile(self.lif_file_path)
            total_images = len(lif_file.image_list)
            
            print(f"📊 Всего образцов в файле: {total_images}")
            print(f"🔍 Анализируем образцы {start_from+1} - {min(start_from+num_samples, total_images)}")
            
            all_vesicle_intensities = []
            
            for idx in range(start_from, min(start_from + num_samples, total_images)):
                image_item = lif_file.image_list[idx]
                
                # Получаем имя образца
                if isinstance(image_item, dict):
                    sample_name = image_item.get('name', f'sample_{idx+1}')
                else:
                    sample_name = getattr(image_item, 'name', f'sample_{idx+1}')
                
                print(f"\n{'='*80}")
                print(f"🔬 ОБРАЗЕЦ {idx+1}/{total_images}: {sample_name}")
                print(f"{'='*80}")
                
                # Загружаем канал везикул (индекс 2)
                channel_data, name = self.load_channel(image_item, channel_idx=2)
                
                if channel_data is None:
                    print(f"   ⚠️ Пропускаем образец (не удалось загрузить канал)")
                    continue
                
                # Сегментируем везикулы
                vesicles_mask, vesicle_stats = self.segment_vesicles(channel_data)
                
                if vesicles_mask is None or len(vesicle_stats) == 0:
                    print(f"   ⚠️ Не найдено везикул в образце")
                    continue
                
                # Создаем визуализацию
                vis_image = self.create_visualization(channel_data, vesicles_mask, vesicle_stats, name)
                
                # Анализируем распределение интенсивностей
                intensity_stats = self.analyze_intensity_distribution(vesicle_stats)
                
                if intensity_stats:
                    all_vesicle_intensities.extend([v['mean_intensity'] for v in vesicle_stats])
                    
                    print(f"\n📊 ИНТЕНСИВНОСТЬ ВЕЗИКУЛ:")
                    print(f"   • Количество: {intensity_stats['count']}")
                    print(f"   • Средняя: {intensity_stats['mean']:.1f}")
                    print(f"   • Медиана: {intensity_stats['median']:.1f}")
                    print(f"   • 25% перцентиль: {intensity_stats['q25']:.1f}")
                    print(f"   • 90% перцентиль: {intensity_stats['q90']:.1f}")
                    print(f"   • Min: {intensity_stats['min']:.1f}")
                    print(f"   • Max: {intensity_stats['max']:.1f}")
                
                # Создаем графики
                self.plot_results(channel_data, vesicle_stats, name)
                
                # Сохраняем результаты
                self.save_results(channel_data, vesicles_mask, vis_image, vesicle_stats, name)
            
            # Общая статистика по всем образцам
            if all_vesicle_intensities:
                print(f"\n{'='*80}")
                print("📊 ОБЩАЯ СТАТИСТИКА ПО ВСЕМ ВЕЗИКУЛАМ")
                print('='*80)
                
                print(f"   • Всего везикул: {len(all_vesicle_intensities)}")
                print(f"   • Средняя интенсивность: {np.mean(all_vesicle_intensities):.1f}")
                print(f"   • Медиана: {np.median(all_vesicle_intensities):.1f}")
                print(f"   • 10% перцентиль: {np.percentile(all_vesicle_intensities, 10):.1f}")
                print(f"   • 25% перцентиль: {np.percentile(all_vesicle_intensities, 25):.1f}")
                print(f"   • 75% перцентиль: {np.percentile(all_vesicle_intensities, 75):.1f}")
                print(f"   • 90% перцентиль: {np.percentile(all_vesicle_intensities, 90):.1f}")
                
                print(f"\n💡 РЕКОМЕНДУЕМЫЙ ПОРОГ ДЛЯ ОТСЕЧЕНИЯ ФОНА:")
                print(f"   • Консервативный: {np.percentile(all_vesicle_intensities, 10):.0f} (сохранит 90% везикул)")
                print(f"   • Сбалансированный: {np.percentile(all_vesicle_intensities, 25):.0f} (сохранит 75% везикул)")
                print(f"   • Агрессивный: {np.percentile(all_vesicle_intensities, 50):.0f} (сохранит 50% везикул)")
                
                # Сохраняем общую статистику
                summary_path = os.path.join(self.output_dir, "summary_report.txt")
                with open(summary_path, 'w', encoding='utf-8') as f:
                    f.write("ОБЩАЯ СТАТИСТИКА ПО ВСЕМ ОБРАЗЦАМ\n")
                    f.write("=" * 60 + "\n\n")
                    f.write(f"Всего везикул: {len(all_vesicle_intensities)}\n")
                    f.write(f"Средняя интенсивность: {np.mean(all_vesicle_intensities):.1f}\n")
                    f.write(f"Медиана: {np.median(all_vesicle_intensities):.1f}\n")
                    f.write(f"10% перцентиль: {np.percentile(all_vesicle_intensities, 10):.1f}\n")
                    f.write(f"25% перцентиль: {np.percentile(all_vesicle_intensities, 25):.1f}\n")
                    f.write(f"75% перцентиль: {np.percentile(all_vesicle_intensities, 75):.1f}\n")
                    f.write(f"90% перцентиль: {np.percentile(all_vesicle_intensities, 90):.1f}\n\n")
                    
                    f.write("РЕКОМЕНДАЦИИ:\n")
                    f.write(f"  • Консервативный порог: {np.percentile(all_vesicle_intensities, 10):.0f}\n")
                    f.write(f"  • Сбалансированный порог: {np.percentile(all_vesicle_intensities, 25):.0f}\n")
                    f.write(f"  • Агрессивный порог: {np.percentile(all_vesicle_intensities, 50):.0f}\n")
                
                print(f"\n📄 Полный отчет сохранен: {summary_path}")
            
            print(f"\n✅ Анализ завершен!")
            print(f"📁 Результаты в папке: {self.output_dir}")
            
        except Exception as e:
            print(f"\n❌ Ошибка обработки: {e}")
            traceback.print_exc()

def main():
    """Главная функция"""
    
    # ⭐ ИЗМЕНИТЕ ЭТИ ПУТИ
    LIF_FILE_PATH = "C:/Users/Petr/VSCode_Python/fibros_analysis/02_raw_data/lif_files/IHC CK 647 EV 594.lif"
    
    # ⭐ НАСТРОЙКИ
    NUM_SAMPLES = 40  # Количество образцов для анализа
    START_FROM = 0   # С какого образца начать (0 = первый)
    
    # Создаем папку с временной меткой
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"vesicle_intensity_test_{timestamp}"
    
    # Запускаем тестер
    tester = VesicleIntensityTester(LIF_FILE_PATH, output_dir)
    tester.process_samples(num_samples=NUM_SAMPLES, start_from=START_FROM)

if __name__ == "__main__":
    main()