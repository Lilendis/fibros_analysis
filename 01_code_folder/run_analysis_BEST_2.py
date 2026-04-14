# run_analysis.py
import os
import sys
from multi_lif_analyzer import MultiSampleLifAnalyzer

def main():
    # === НАСТРОЙКИ ===
    LIF_FILE_PATH = "C:/Users/Petr/VSCode_Python/fibros_analysis/02_raw_data/lif_files/aSMa_647_EVs_594.lif"
    IGG_LIF_FILE_PATH = "C:/Users/Petr/VSCode_Python/fibros_analysis/02_raw_data/lif_files/IHC_IgG_647_EVs_594.lif"
    OUTPUT_DIR = "C:/Users/Petr/VSCode_Python/fibros_analysis/03_processing_results"
    
    ANALYSIS_CONFIG = {
        'min_vesicle_size': 1,
        'max_vesicle_size': 200,
        'collagen_percentile': 90,
        'protein_subtraction_percent': 90,
        'vesicle_brightness_enhancement': True,
        'vesicles_contrast_factor': 1.0,  # ⭐ НОВЫЙ ПАРАМЕТР
        'protein_contrast_factor': 2.0,   # ⭐ НОВЫЙ ПАРАМЕТР
        'exclude_patterns': ['igg', 'pbs', 'control'],  # ⭐ НОВЫЙ ПАРАМЕТР
    }
    
    print("=" * 60)
    print("АВТОМАТИЧЕСКИЙ АНАЛИЗ ФИБРОЗА ЛЕГКИХ")
    print("С УЛУЧШЕННОЙ ВИЗУАЛИЗАЦИЕЙ И КОРРЕКЦИЕЙ")
    print("=" * 60)
    
    # Проверка существования основных файлов
    if not os.path.exists(LIF_FILE_PATH):
        error_msg = f"❌ ОШИБКА: Основной файл не найден: {LIF_FILE_PATH}"
        print(error_msg)
        return
    
    if not os.path.exists(IGG_LIF_FILE_PATH):
        error_msg = f"❌ ОШИБКА: IgG файл не найден: {IGG_LIF_FILE_PATH}"
        print(error_msg)
        return
    
    # Проверка существования выходной директории
    if not os.path.exists(OUTPUT_DIR):
        try:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            print(f"📁 Создана выходная папка: {OUTPUT_DIR}")
        except Exception as e:
            error_msg = f"❌ ОШИБКА: Не удалось создать выходную папку {OUTPUT_DIR}: {e}"
            print(error_msg)
            return
    
    # Создание анализатора
    analyzer = MultiSampleLifAnalyzer(
        min_vesicle_size=ANALYSIS_CONFIG['min_vesicle_size'],
        max_vesicle_size=ANALYSIS_CONFIG['max_vesicle_size'],
        vesicles_contrast_factor=ANALYSIS_CONFIG['vesicles_contrast_factor'],  # ⭐ НОВЫЙ ПАРАМЕТР
        protein_contrast_factor=ANALYSIS_CONFIG['protein_contrast_factor']     # ⭐ НОВЫЙ ПАРАМЕТР
    )
    
    print("\n🔬 ЗАПУСК АНАЛИЗА С УЛУЧШЕННОЙ ВИЗУАЛИЗАЦИЕЙ:")
    print("   ✅ Задача 1: Автоматическое определение IgG фона для белка")
    print("   ✅ Задача 2: Объединение данных из IgG файла и основного файла") 
    print("   ✅ Задача 3: Вычитание 90% IgG фона из целевых образцов")
    print("   ✅ Задача 4: Интерактивная коррекция везикул по коллагену")
    print("   ✅ Задача 5: Настраиваемый контраст для везикул и белка")  # ⭐ ОБНОВЛЕНО
    print("   ✅ Задача 6: Группировка по времени (30min/3h)")  # ⭐ НОВАЯ ЗАДАЧА
    print("   ✅ Задача 7: Исключение IgG и PBS образцов из статистики")  # ⭐ НОВАЯ ЗАДАЧА
    print("   ✅ Задача 8: Анализ везикул в клетках")  # ⭐ НОВАЯ ЗАДАЧА
    print(f"📁 Основной файл: {LIF_FILE_PATH}")
    print(f"📁 IgG файл: {IGG_LIF_FILE_PATH}")
    print(f"📁 Выходная папка: {OUTPUT_DIR}")
    
    print("\n🔧 НАСТРОЙКИ АНАЛИЗА:")
    print(f"   • Перцентиль коллагена: {ANALYSIS_CONFIG['collagen_percentile']}%")
    print(f"   • Вычитание IgG фона для белка: {ANALYSIS_CONFIG['protein_subtraction_percent']}%")
    print(f"   • Коэффициент контраста везикул: {ANALYSIS_CONFIG['vesicles_contrast_factor']}")  # ⭐ НОВАЯ НАСТРОЙКА
    print(f"   • Коэффициент контраста белка: {ANALYSIS_CONFIG['protein_contrast_factor']}")    # ⭐ НОВАЯ НАСТРОЙКА
    print(f"   • Минимальный размер везикул: {ANALYSIS_CONFIG['min_vesicle_size']} px")
    print(f"   • Максимальный размер везикул: {ANALYSIS_CONFIG['max_vesicle_size']} px")
    print(f"   • Исключаемые образцы: {', '.join(ANALYSIS_CONFIG['exclude_patterns'])}")  # ⭐ НОВАЯ НАСТРОЙКА
    print("-" * 60)
    
    try:
        # Сначала загружаем IgG данные
        print("\n🔍 ЗАГРУЗКА IgG ДАННЫХ...")
        print("   📖 Анализ гистограмм IgG образцов для определения фона белка...")
        igg_loaded = analyzer.load_igg_from_file(IGG_LIF_FILE_PATH, OUTPUT_DIR)
        
        if not igg_loaded:
            error_msg = "❌ ОШИБКА: Не удалось загрузить IgG данные. Анализ прерван."
            print(error_msg)
            return
        
        print("✅ IgG данные успешно загружены и проанализированы!")
        
        # Затем обрабатываем основной файл
        print("\n🔍 ОБРАБОТКА ОСНОВНОГО ФАЙЛА...")
        print("   🔎 Поиск IgG образцов в основном файле...")
        print("   💡 Для каждого образца будет предложено ввести процент вычитания везикул")
        print("   💡 Нажмите Enter для использования рекомендованного значения")
        print("   💡 Введите 'skip' для пропуска коррекции везикул")
        print("   💡 Коррекция белка выполняется автоматически")
        print("   💡 Образцы группируются по времени: 30min и 3h")  # ⭐ НОВАЯ ИНФОРМАЦИЯ
        print("   💡 IgG и PBS образцы исключаются из статистики")  # ⭐ НОВАЯ ИНФОРМАЦИЯ
        print("-" * 60)
        
        result = analyzer.process_lif_file(LIF_FILE_PATH, OUTPUT_DIR)
        
        if result and result[0] is not None:
            summaries, output_dir = result
            print("\n" + "=" * 60)
            print("✅ АНАЛИЗ УСПЕШНО ЗАВЕРШЕН!")
            print("=" * 60)
            print(f"📁 Результаты сохранены в: {output_dir}")
            
            # Фильтрация статистики (исключаем IgG и PBS)
            filtered_summaries = [s for s in summaries if not any(pattern in s['sample_name'].lower() for pattern in ANALYSIS_CONFIG['exclude_patterns'])]
            excluded_count = len(summaries) - len(filtered_summaries)
            
            print(f"📊 Обработано образцов: {len(summaries)}")
            if excluded_count > 0:
                print(f"📊 Исключено из статистики: {excluded_count} (IgG/PBS)")
                print(f"📊 Использовано для анализа: {len(filtered_summaries)}")
            
            # Дополнительная статистика
            if filtered_summaries:
                total_vesicles = sum(s['total_vesicles'] for s in filtered_summaries)
                colocalized_vesicles = sum(s['colocalized_vesicles'] for s in filtered_summaries)
                vesicles_in_cells = sum(s.get('vesicles_in_cells', 0) for s in filtered_summaries)  # ⭐ НОВАЯ СТАТИСТИКА
                avg_percentage = sum(s['colocalization_percentage'] for s in filtered_summaries) / len(filtered_summaries)
                avg_in_cells = vesicles_in_cells / len(filtered_summaries) if filtered_summaries else 0  # ⭐ НОВАЯ СТАТИСТИКА
                
                print(f"📈 СТАТИСТИКА КОЛОКАЛИЗАЦИИ:")
                print(f"   • Всего везикул: {total_vesicles}")
                print(f"   • Везикул в макрофагах: {colocalized_vesicles}")
                print(f"   • Везикул в клетках: {vesicles_in_cells}")  # ⭐ НОВАЯ СТАТИСТИКА
                print(f"   • Средний процент колокализации: {avg_percentage:.2f}%")
                print(f"   • Среднее везикул в клетках на образец: {avg_in_cells:.1f}")  # ⭐ НОВАЯ СТАТИСТИКА
                
                if total_vesicles > 0:
                    colocalization_rate = (colocalized_vesicles / total_vesicles) * 100
                    in_cells_rate = (vesicles_in_cells / total_vesicles) * 100
                    print(f"   • Процент везикул в макрофагах: {colocalization_rate:.2f}%")
                    print(f"   • Процент везикул в клетках: {in_cells_rate:.2f}%")  # ⭐ НОВАЯ СТАТИСТИКА
            
            # Информация о коррекции белка
            if hasattr(analyzer, 'protein_igg_background'):
                print(f"\n🎯 ИНФОРМАЦИЯ О КОРРЕКЦИИ БЕЛКА:")
                print(f"   • Определенный IgG фон: {analyzer.protein_igg_background:.2f}")
                print(f"   • Вычтено из целевых образцов: {analyzer.protein_igg_background * 0.9:.2f} (90%)")
            
            print("\n📁 СОХРАНЕННЫЕ ФАЙЛЫ ДЛЯ КАЖДОГО ОБРАЗЦА:")
            print("   • [sample]_composite.png - основной композит")
            print("   • [sample]_composite_annotated.png - композит с аннотациями")
            print("   • [sample]_channel2_vesicles_ORIGINAL.png - исходные везикулы")
            print("   • [sample]_channel2_vesicles_CORRECTED.png - скорректированные везикулы") 
            print("   • [sample]_channel2_vesicles_annotated.png - везикулы с кругами")
            print("   • [sample]_channel3_protein_ORIGINAL.png - исходный белок")
            print("   • [sample]_channel3_protein_CORRECTED.png - скорректированный белок")
            print("   • [sample]_channel0_nuclei.png - ядра")
            print("   • [sample]_channel1_collagen.png - коллаген (если есть)")
            
            print("\n💡 РЕКОМЕНДАЦИИ:")
            print("   • Сравните *_ORIGINAL.png и *_CORRECTED.png для оценки коррекции")
            print("   • Настройте коэффициенты контраста в ANALYSIS_CONFIG при необходимости")
            print("   • Проверьте статистику везикул в клетках для анализа проникновения")
            print("=" * 60)
            
        else:
            error_msg = result[1] if result else "❌ ОШИБКА: Неизвестная ошибка при обработке"
            print(f"\n{error_msg}")
            
    except KeyboardInterrupt:
        print(f"\n⏹️  Анализ прерван пользователем")
        
    except Exception as e:
        error_msg = f"❌ ОШИБКА ПРИ ОБРАБОТКЕ: {str(e)}"
        print(f"\n{error_msg}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()