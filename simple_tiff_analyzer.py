# simple_tiff_analyzer.py
import numpy as np
import cv2
import os
import glob
from skimage import exposure, filters, measure, morphology, segmentation
from scipy import ndimage
import pandas as pd
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from datetime import datetime
import time
import tifffile

class SimpleTiffAnalyzer:
    def __init__(self, min_vesicle_size=5, max_vesicle_size=100):
        self.min_vesicle_size = min_vesicle_size
        self.max_vesicle_size = max_vesicle_size
    
    def find_and_group_tiff_files(self, input_folder):
        """Находит и группирует TIFF файлы по образцам"""
        print(f"🔍 Поиск TIFF файлов в: {input_folder}")
        
        tiff_files = glob.glob(os.path.join(input_folder, "**", "*.tif"), recursive=True)
        tiff_files.extend(glob.glob(os.path.join(input_folder, "**", "*.tiff"), recursive=True))
        
        if not tiff_files:
            print("❌ TIFF файлы не найдены")
            return {}
        
        print(f"📁 Найдено TIFF файлов: {len(tiff_files)}")
        
        # Группируем по именам (предполагаем, что файлы называются sample_X_channel_Y.tif)
        samples = {}
        
        for tiff_file in tiff_files:
            filename = os.path.basename(tiff_file)
            print(f"   📄 {filename}")
            
            # Простая логика группировки - по общему префиксу
            if 'channel' in filename.lower():
                # sample_1_channel_1.tif -> sample_1
                base_name = filename.split('_channel')[0]
            else:
                base_name = os.path.splitext(filename)[0]
            
            if base_name not in samples:
                samples[base_name] = []
            
            samples[base_name].append(tiff_file)
        
        print(f"📊 Образцов найдено: {len(samples)}")
        return samples
    
    def load_tiff_file(self, filepath):
        """Загружает TIFF файл"""
        try:
            img = tifffile.imread(filepath)
            if img.dtype != np.uint8:
                img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            return img
        except:
            return None
    
    def process_tiff_files(self, input_folder, output_base_dir):
        """Обрабатывает все TIFF файлы"""
        start_time = time.time()
        
        # Создаем папку для результатов
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(output_base_dir, f"analysis_tiff_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"🚀 НАЧАЛО АНАЛИЗА TIFF ФАЙЛОВ")
        print(f"📁 Входная папка: {input_folder}")
        print(f"📁 Выходная папка: {output_dir}")
        print("=" * 60)
        
        # Находим файлы
        samples = self.find_and_group_tiff_files(input_folder)
        
        if not samples:
            print("❌ Не найдено файлов для анализа")
            return
        
        all_summaries = []
        
        for sample_name, file_list in samples.items():
            print(f"\n🔬 ОБРАЗЕЦ: {sample_name}")
            print(f"   📊 Файлов: {len(file_list)}")
            
            # Создаем папку для образца
            sample_dir = os.path.join(output_dir, sample_name)
            os.makedirs(sample_dir, exist_ok=True)
            
            # Загружаем каналы
            channels = {}
            for i, filepath in enumerate(file_list[:4]):  # Берем первые 4 файла как каналы
                print(f"   📥 Загрузка канала {i}...")
                channel_data = self.load_tiff_file(filepath)
                if channel_data is not None:
                    channels[i] = channel_data
                    print(f"   ✅ Канал {i} загружен: {channel_data.shape}")
            
            if len(channels) < 3:
                print(f"   ❌ Недостаточно каналов ({len(channels)}), нужно 3")
                continue
            
            # Создаем композит
            composite = self.create_composite(channels, sample_dir, sample_name)
            if composite is None:
                continue
            
            # Анализируем колокализацию
            summary = self.analyze_sample(channels, sample_dir, sample_name)
            if summary:
                all_summaries.append(summary)
        
        # Сохраняем сводку
        if all_summaries:
            summary_df = pd.DataFrame(all_summaries)
            summary_path = os.path.join(output_dir, "analysis_summary.csv")
            summary_df.to_csv(summary_path, index=False)
            
            processing_time = time.time() - start_time
            print(f"\n✅ АНАЛИЗ ЗАВЕРШЕН!")
            print(f"📊 Обработано образцов: {len(all_summaries)}")
            print(f"⏱️ Время: {processing_time:.1f} сек")
            print(f"📁 Результаты: {output_dir}")
        
        return all_summaries, output_dir
    
    def create_composite(self, channels, output_dir, sample_name):
        """Создает композитное изображение"""
        try:
            # Предобработка
            red = self.preprocess_channel(channels[0], 0)
            blue = self.preprocess_channel(channels[1], 1)
            yellow = self.preprocess_channel(channels[2], 2)
            
            # Создаем RGB
            composite = np.zeros((red.shape[0], red.shape[1], 3), dtype=np.uint8)
            composite[:,:,2] = red    # Красный - везикулы
            composite[:,:,0] = blue   # Синий - ядра
            composite[:,:,1] = yellow # Зеленый - макрофаги
            
            # Сохраняем
            output_path = os.path.join(output_dir, f"{sample_name}_composite.png")
            cv2.imwrite(output_path, composite)
            print(f"   ✅ Композит сохранен: {output_path}")
            
            return composite
        except Exception as e:
            print(f"   ❌ Ошибка создания композита: {e}")
            return None
    
    def preprocess_channel(self, img, channel_type):
        """Предобработка канала"""
        img_denoised = cv2.medianBlur(img, 3)
        try:
            threshold = filters.threshold_otsu(img_denoised)
            background_reduced = cv2.subtract(img_denoised, int(threshold * 0.7))
            background_reduced = np.clip(background_reduced, 0, 255)
        except:
            background_reduced = img_denoised
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_contrast = clahe.apply(background_reduced)
        
        gamma_values = {0: 0.7, 1: 0.8, 2: 0.8, 3: 0.9}
        gamma = gamma_values.get(channel_type, 0.8)
        img_enhanced = exposure.adjust_gamma(img_contrast, gamma=gamma)
        
        return img_enhanced
    
    def analyze_sample(self, channels, output_dir, sample_name):
        """Анализирует один образец"""
        try:
            # Сегментация
            vesicles_binary = self.segment_vesicles(channels[0])
            macrophages_labels = self.segment_macrophages(channels[2])
            
            if vesicles_binary is None or macrophages_labels is None:
                return None
            
            # Анализ колокализации
            vesicles_labels = measure.label(vesicles_binary)
            vesicles_props = measure.regionprops(vesicles_labels)
            
            total_vesicles = len(vesicles_props)
            colocalized_vesicles = 0
            
            for vesicle_props in vesicles_props:
                vesicle_coords = vesicle_props.coords
                for coord in vesicle_coords:
                    y, x = coord
                    if y < macrophages_labels.shape[0] and x < macrophages_labels.shape[1]:
                        if macrophages_labels[y, x] > 0:
                            colocalized_vesicles += 1
                            break
            
            percentage = (colocalized_vesicles / total_vesicles * 100) if total_vesicles > 0 else 0
            
            summary = {
                'sample_name': sample_name,
                'total_vesicles': total_vesicles,
                'colocalized_vesicles': colocalized_vesicles,
                'colocalization_percentage': percentage
            }
            
            print(f"   📊 Результаты: {total_vesicles} везикул, {colocalized_vesicles} в макрофагах ({percentage:.1f}%)")
            
            return summary
            
        except Exception as e:
            print(f"   ❌ Ошибка анализа: {e}")
            return None
    
    def segment_vesicles(self, channel):
        """Сегментация везикул"""
        try:
            binary = cv2.adaptiveThreshold(channel, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            binary_bool = binary.astype(bool)
            cleaned = morphology.remove_small_objects(binary_bool, min_size=self.min_vesicle_size)
            cleaned = morphology.remove_large_objects(cleaned, max_size=self.max_vesicle_size)
            
            return cleaned.astype(np.uint8) * 255
        except:
            return None
    
    def segment_macrophages(self, channel):
        """Сегментация макрофагов"""
        try:
            blurred = cv2.GaussianBlur(channel, (5, 5), 0)
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            binary = ndimage.binary_fill_holes(binary)
            
            distance = ndimage.distance_transform_edt(binary)
            local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)), labels=binary)
            markers = measure.label(local_maxi)
            labels = segmentation.watershed(-distance, markers, mask=binary)
            
            return labels
        except:
            return None

def main():
    """Главная функция"""
    print("============================================================")
    print("ПРОСТОЙ АНАЛИЗ TIFF ФАЙЛОВ")
    print("============================================================")
    
    TIFF_FOLDER = r"C:/Users/Petr/VSCode_Python/fibros_analysis/02_raw_data/tiff_files"
    OUTPUT_DIR = r"C:/Users/Petr/VSCode_Python/fibros_analysis/03_processing_results"
    
    print(f"🔬 Запуск анализа TIFF...")
    print(f"📁 Входная папка: {TIFF_FOLDER}")
    print(f"📁 Выходная папка: {OUTPUT_DIR}")
    print("-" * 60)
    
    analyzer = SimpleTiffAnalyzer()
    summaries, output_dir = analyzer.process_tiff_files(TIFF_FOLDER, OUTPUT_DIR)
    
    if summaries:
        print(f"\n✅ АНАЛИЗ УСПЕШНО ЗАВЕРШЕН!")
        print(f"📁 Результаты: {output_dir}")

if __name__ == "__main__":
    main()