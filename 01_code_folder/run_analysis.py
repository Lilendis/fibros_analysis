# run_analysis.py
import os
import sys
from datetime import datetime
import traceback
import numpy as np

try:
    from multi_lif_analyzer import MultiSampleLifAnalyzer
except ImportError as e:
    print(f"❌ ОШИБКА: Не удалось импортировать MultiSampleLifAnalyzer")
    print(f"   Убедитесь, что файл multi_lif_analyzer.py находится в той же папке")
    print(f"   Ошибка: {e}")
    sys.exit(1)

def main():
    # === НАСТРОЙКИ ===
    # ⭐ ИЗМЕНИТЕ ЭТИ ПУТИ ПОД ВАШУ СИСТЕМУ
    LIF_FILE_PATH = "C:/Users/Petr/VSCode_Python/fibros_analysis/02_raw_data/lif_files/FAPa 647 EVs 594 (1).lif"
    IGG_LIF_FILE_PATH = "C:/Users/Petr/VSCode_Python/fibros_analysis/02_raw_data/lif_files/IHC_IgG_647_EVs_594.lif"
    OUTPUT_DIR = "C:/Users/Petr/VSCode_Python/fibros_analysis/03_processing_results"
    
    # ⭐ ПУТЬ ДЛЯ СОХРАНЕНИЯ IgG БАЗЫ ДАННЫХ (ОБЩИЙ ДЛЯ ВСЕХ АНАЛИЗОВ)
    IGG_DATABASE_PATH = "C:/Users/Petr/VSCode_Python/fibros_analysis/02_raw_data/igg_database"

    ANALYSIS_CONFIG = {
        'min_vesicle_size': 1,
        'max_vesicle_size': 200,
        'collagen_percentile': 80,
        'vesicle_subtraction_factor': 0.8,
        'protein_subtraction_percent': 90,
        'vesicles_contrast_factor': 1.0,
        'protein_brightness_factor': 1.7,
        'protein_contrast_factor': 1.5,
        'exclude_patterns': ['igg', 'pbs', 'control', 'neg'],
        'split_large_clusters': True,
        'intensity_ratio_threshold': 1.1,
        'intensity_diff_threshold': 10,
        'background_method': 'original',
        'igg_database_path': IGG_DATABASE_PATH,
        'min_vesicle_intensity': 100,


        # ⭐ НОВЫЕ ПАРАМЕТРЫ
        'channel_background_percentile': 95,
        'background_threshold_percentile': 95,
        'igg_fallback_percentile': 90,
        'default_protein_background': 50.0,
        'nuclei_gamma': 0.8,
        'vesicles_gamma': 1.0,
        'protein_gamma': 0.8,
        
    }
    
    print("=" * 70)
    print("АВТОМАТИЧЕСКИЙ АНАЛИЗ КОЛОКАЛИЗАЦИИ ВЕЗИКУЛ И МАКРОФАГОВ")
    print("С УЛУЧШЕННОЙ СИСТЕМОЙ IgG КОРРЕКЦИИ И БАЗОЙ ДАННЫХ")
    print("=" * 70)
    
    print(f"🕒 Дата и время запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Проверка существования основных файлов
    if not os.path.exists(LIF_FILE_PATH):
        error_msg = f"❌ ОШИБКА: Основной файл не найден: {LIF_FILE_PATH}"
        print(error_msg)
        print(f"   Проверьте путь и убедитесь, что файл существует")
        return
    
    # ⭐ IgG файл теперь НЕОБЯЗАТЕЛЬНЫЙ
    if IGG_LIF_FILE_PATH:
        if os.path.exists(IGG_LIF_FILE_PATH):
            print(f"✅ IgG файл найден: {IGG_LIF_FILE_PATH}")
        else:
            print(f"⚠️  IgG файл не найден: {IGG_LIF_FILE_PATH}")
            print(f"   Анализ будет использовать исторические данные и IgG из основного файла")
    else:
        print("ℹ️  IgG файл не указан, используем только базу данных")
    
    # Создаем выходную директорию если не существует
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Создаем директорию для базы данных IgG если не существует
    if ANALYSIS_CONFIG['igg_database_path']:
        os.makedirs(ANALYSIS_CONFIG['igg_database_path'], exist_ok=True)
        print(f"📁 Директория базы IgG данных: {ANALYSIS_CONFIG['igg_database_path']}")
    
    # Создание анализатора с базой данных IgG
        # Создание анализатора с адаптивной системой
    try:
        analyzer = MultiSampleLifAnalyzer(
            min_vesicle_size=ANALYSIS_CONFIG['min_vesicle_size'],
            max_vesicle_size=ANALYSIS_CONFIG['max_vesicle_size'],
            background_method=ANALYSIS_CONFIG['background_method'],
            subtraction_factor=ANALYSIS_CONFIG['vesicle_subtraction_factor'],
            vesicles_contrast_factor=ANALYSIS_CONFIG['vesicles_contrast_factor'],
            protein_contrast_factor=ANALYSIS_CONFIG['protein_contrast_factor'],
            protein_brightness_factor=ANALYSIS_CONFIG['protein_brightness_factor'],
            split_large_clusters=ANALYSIS_CONFIG['split_large_clusters'],
            intensity_ratio_threshold=ANALYSIS_CONFIG['intensity_ratio_threshold'],
            intensity_diff_threshold=ANALYSIS_CONFIG['intensity_diff_threshold'],
            save_igg_data_path=ANALYSIS_CONFIG['igg_database_path'],
            
            # ⭐ ПЕРЕДАЧА НОВЫХ ПАРАМЕТРОВ
            protein_subtraction_percent=ANALYSIS_CONFIG['protein_subtraction_percent'],
            collagen_percentile=ANALYSIS_CONFIG['collagen_percentile'],
            exclude_patterns=ANALYSIS_CONFIG['exclude_patterns'],
            channel_background_percentile=ANALYSIS_CONFIG['channel_background_percentile'],
            background_threshold_percentile=ANALYSIS_CONFIG['background_threshold_percentile'],
            igg_fallback_percentile=ANALYSIS_CONFIG['igg_fallback_percentile'],
            default_protein_background=ANALYSIS_CONFIG['default_protein_background'],
            nuclei_gamma=ANALYSIS_CONFIG['nuclei_gamma'],
            vesicles_gamma=ANALYSIS_CONFIG['vesicles_gamma'],
            protein_gamma=ANALYSIS_CONFIG['protein_gamma'],
            min_vesicle_intensity=ANALYSIS_CONFIG['min_vesicle_intensity'],
            
        )
        print("✅ Анализатор успешно создан")
        print("   ⭐ Адаптивная система порогов: ВКЛЮЧЕНА")
    except Exception as e:
        print(f"❌ Ошибка создания анализатора: {e}")
        traceback.print_exc()
        return
    
    print("\n🔬 ЗАПУСК АНАЛИЗА С УЛУЧШЕННОЙ IgG СИСТЕМОЙ:")
    print("   ✅ Задача 1: Автоматическое определение IgG фона из всех источников")
    print("   ✅ Задача 2: База данных IgG интенсивностей для всех анализов")
    print("   ✅ Задача 3: Использование IgG из основного файла при отсутствии отдельного")
    print("   ✅ Задача 4: Резервная стратегия для образцов без IgG данных")
    print("   ✅ Задача 5: Сохранение оригинальных каналов для сравнения")
    print("   ✅ Задача 6: Сегментация по локальным максимумам")
    print("   ✅ Задача 7: Анализ везикул в клетках")
    print("   ✅ Задача 8: Автоматическое исключение IgG/PBS образцов")
    print(f"\n📁 Основной файл: {os.path.basename(LIF_FILE_PATH)}")
    print(f"📁 IgG файл: {os.path.basename(IGG_LIF_FILE_PATH) if IGG_LIF_FILE_PATH and os.path.exists(IGG_LIF_FILE_PATH) else 'не найден'}")
    print(f"📁 Выходная папка: {OUTPUT_DIR}")
    
    print("\n🔧 НАСТРОЙКИ АНАЛИЗА:")
    print(f"   • IgG база данных: {ANALYSIS_CONFIG['igg_database_path']}")
    print(f"   • Процент вычитания IgG фона: {ANALYSIS_CONFIG['protein_subtraction_percent']}%")
    print(f"   • Фактор вычитания фона везикул: {ANALYSIS_CONFIG['vesicle_subtraction_factor']}")
    print(f"   • Метод вычитания фона: {ANALYSIS_CONFIG['background_method']}")
    print(f"   • Коэффициент контраста везикул: {ANALYSIS_CONFIG['vesicles_contrast_factor']}")
    print(f"   • Коэффициент контраста белка: {ANALYSIS_CONFIG['protein_contrast_factor']}")
    print(f"   • Коэффициент яркости белка: {ANALYSIS_CONFIG['protein_brightness_factor']}")
    print(f"   • Минимальный размер везикул: {ANALYSIS_CONFIG['min_vesicle_size']} px")
    print(f"   • Максимальный размер везикул: {ANALYSIS_CONFIG['max_vesicle_size']} px")
    print(f"   • Порог отношения интенсивностей: {ANALYSIS_CONFIG['intensity_ratio_threshold']}")
    print(f"   • Порог разности интенсивностей: {ANALYSIS_CONFIG['intensity_diff_threshold']}")
    print(f"   • Разделение скоплений: {'ВКЛ' if ANALYSIS_CONFIG['split_large_clusters'] else 'ВЫКЛ'}")
    print(f"   • Исключаемые образцы: {', '.join(ANALYSIS_CONFIG['exclude_patterns'])}")
    print("-" * 70)
    
    try:
        # Сначала загружаем IgG данные (если файл существует)
        if IGG_LIF_FILE_PATH and os.path.exists(IGG_LIF_FILE_PATH):
            print("\n🔍 ЗАГРУЗКА IgG ДАННЫХ ИЗ ФАЙЛА...")
            igg_loaded = analyzer.load_igg_from_file(IGG_LIF_FILE_PATH, OUTPUT_DIR)
            
            if igg_loaded:
                print("✅ IgG данные из файла успешно загружены и добавлены в базу!")
            else:
                print("⚠️  Не удалось загрузить IgG данные из файла")
                print("   Продолжаем анализ с существующими данными")
        else:
            print("\n📊 ИСПОЛЬЗУЕМ СУЩЕСТВУЮЩУЮ БАЗУ IgG ДАННЫХ")
            if hasattr(analyzer, 'igg_database') and analyzer.igg_database.get('protein_igg_background'):
                bg = analyzer.igg_database['protein_igg_background']
                count = analyzer.igg_database.get('samples_count', 0)
                print(f"   ✅ Загружен IgG фон из базы: {bg:.2f}")
                print(f"   📊 На основе {count} исторических образцов")
            else:
                print("   ℹ️  База данных IgG пуста или не найдена")
                print("   Система будет использовать IgG образцы из основного файла")
        
        # Затем обрабатываем основной файл
        print("\n🔍 ОБРАБОТКА ОСНОВНОГО ФАЙЛА...")
        print("   🔎 Поиск IgG образцов в основном файле...")
        print("   💡 СИСТЕМА IgG КОРРЕКЦИИ:")
        print("      • Используются ВСЕ доступные источники IgG")
        print("      • Данные сохраняются для будущих анализов")
        print("      • При отсутствии IgG - используется среднее значение")
        print("      • База данных автоматически обновляется")
        print("   💡 ОСОБЕННОСТИ АНАЛИЗА:")
        print("      • Везикулы сегментируются по локальным максимумам")
        print("      • Большие скопления разделяются автоматически")
        print("      • Сохраняются оригинальные и скорректированные каналы")
        print("      • Для каждого образца будет предложено вычитание везикул")
        print("-" * 70)
        
        result = analyzer.process_lif_file(LIF_FILE_PATH, OUTPUT_DIR)
        
        if result and result[0] is not None:
            summaries, output_dir = result
            # ⭐ ИНФОРМАЦИЯ О АДАПТИВНОЙ ОБРАБОТКЕ
            print(f"\n🎯 АДАПТИВНАЯ ОБРАБОТКА:")
            
            if hasattr(analyzer, 'adaptive_statistics') and analyzer.adaptive_statistics:
                stats = analyzer.adaptive_statistics
                
                # Анализируем статистики адаптации
                thresholds = [s['adaptive_threshold'] for s in stats.values()]
                vesicles_counts = [s['total_vesicles'] for s in stats.values() if 'total_vesicles' in s]
                
                if thresholds:
                    print(f"   • Средний адаптивный порог: {np.mean(thresholds):.1f}")
                    print(f"   • Диапазон порогов: {min(thresholds):.1f} - {max(thresholds):.1f}")
                    
                    # Классифицируем образцы по порогам
                    dark_samples = [name for name, s in stats.items() if s.get('adaptive_threshold', 0) < 20]
                    bright_samples = [name for name, s in stats.items() if s.get('adaptive_threshold', 0) > 35]
                    
                    if dark_samples:
                        print(f"   • Тусклых образцов (<20): {len(dark_samples)}")
                    if bright_samples:
                        print(f"   • Ярких образцов (>35): {len(bright_samples)}")
                
                if vesicles_counts:
                    print(f"   • Среднее везикул на образец: {np.mean(vesicles_counts):.1f}")
                    print(f"   • Диапазон везикул: {min(vesicles_counts)} - {max(vesicles_counts)}")
                
                # Сохраняем сводную статистику адаптации
                try:
                    adaptive_summary_path = os.path.join(output_dir, "00_adaptive_summary.json")
                    import json
                    with open(adaptive_summary_path, 'w', encoding='utf-8') as f:
                        json.dump({
                            'adaptive_statistics': stats,
                            'summary': {
                                'total_samples': len(stats),
                                'mean_threshold': float(np.mean(thresholds)) if thresholds else 0,
                                'min_threshold': float(min(thresholds)) if thresholds else 0,
                                'max_threshold': float(max(thresholds)) if thresholds else 0,
                                'samples_count': len(thresholds)
                            },
                            'config': ANALYSIS_CONFIG,
                            'processing_date': datetime.now().isoformat()
                        }, f, indent=4, ensure_ascii=False)
                    
                    print(f"   • Статистики адаптации сохранены: {adaptive_summary_path}")
                except Exception as e:
                    print(f"   ⚠️  Не удалось сохранить статистики адаптации: {e}")
            else:
                print(f"   ⚠️  Статистики адаптации недоступны")
            print("\n" + "=" * 70)
            print("✅ АНАЛИЗ УСПЕШНО ЗАВЕРШЕН!")
            print("=" * 70)
            
            # Фильтрация статистики (исключаем IgG и PBS образцы)
            filtered_summaries = []
            excluded_samples = []
            
            for summary in summaries:
                sample_name_lower = summary['sample_name'].lower()
                exclude = any(pattern in sample_name_lower for pattern in ANALYSIS_CONFIG['exclude_patterns'])
                
                if exclude:
                    excluded_samples.append(summary['sample_name'])
                else:
                    filtered_summaries.append(summary)
            
            print(f"\n📊 ОБРАБОТКА ДАННЫХ:")
            print(f"   • Всего обработано образцов: {len(summaries)}")
            print(f"   • Исключено (IgG/PBS/Control): {len(excluded_samples)}")
            print(f"   • Использовано для анализа: {len(filtered_summaries)}")
            
            if excluded_samples:
                print(f"   • Исключенные образцы: {', '.join(excluded_samples[:5])}")
                if len(excluded_samples) > 5:
                    print(f"     ... и еще {len(excluded_samples) - 5} образцов")
            
            # ⭐ ИНФОРМАЦИЯ О IgG СИСТЕМЕ
            print(f"\n🎯 СИСТЕМА IgG КОРРЕКЦИИ:")
            
            # Получаем текущий фон
            current_bg = analyzer.get_protein_igg_background()
            
            if hasattr(analyzer, 'igg_database'):
                db = analyzer.igg_database
                total_samples = db.get('samples_count', 0)
                last_update = db.get('last_update', 'неизвестно')
                
                print(f"   • Текущий IgG фон: {current_bg:.2f}")
                print(f"   • Образцов в базе: {total_samples}")
                if last_update != 'неизвестно':
                    try:
                        from datetime import datetime as dt
                        update_time = dt.fromisoformat(last_update.replace('Z', '+00:00'))
                        print(f"   • Последнее обновление: {update_time.strftime('%Y-%m-%d %H:%M')}")
                    except:
                        print(f"   • Последнее обновление: {last_update}")
                print(f"   • Используемый перцентиль: 90%")
            
            # Сохраняем финальную версию базы данных
            if hasattr(analyzer, 'save_igg_database'):
                try:
                    analyzer.save_igg_database()
                    print(f"   • База данных сохранена: {ANALYSIS_CONFIG['igg_database_path']}")
                except Exception as e:
                    print(f"   ⚠️  Не удалось сохранить базу данных: {e}")
            
            # Статистика по фильтрованным образцам
            # Статистика по фильтрованным образцам
            if filtered_summaries:
                print(f"\n📈 СТАТИСТИКА КОЛОКАЛИЗАЦИИ (только целевые образцы):")
                
                total_vesicles = sum(s['total_vesicles'] for s in filtered_summaries)
                vesicles_in_cells = sum(s.get('vesicles_in_cells', 0) for s in filtered_summaries)
                
                # ⭐ ВАЖНО: colocalized_vesicles - это КОЛИЧЕСТВО везикул, колокализованных с белком
                # Оно НЕ может быть больше total_vesicles!
                colocalized_vesicles = sum(s.get('colocalized_vesicles', 0) for s in filtered_summaries)
                
                # Вычисляем средние значения
                n_samples = len(filtered_summaries)
                avg_total = total_vesicles / n_samples if n_samples > 0 else 0
                avg_in_cells = vesicles_in_cells / n_samples if n_samples > 0 else 0
                avg_colocalized = colocalized_vesicles / n_samples if n_samples > 0 else 0
                
                # Проценты (от общего количества везикул)
                if total_vesicles > 0:
                    in_cells_rate = (vesicles_in_cells / total_vesicles) * 100
                    colocalization_rate = (colocalized_vesicles / total_vesicles) * 100
                else:
                    in_cells_rate = 0
                    colocalization_rate = 0
                
                # Средний процент по образцам (не суммарный)
                avg_percentage = sum(s['colocalization_percentage'] for s in filtered_summaries) / n_samples if n_samples > 0 else 0
                avg_percent_in_cells = sum(s.get('percent_in_cells', 0) for s in filtered_summaries) / n_samples if n_samples > 0 else 0
                
                print(f"   📊 АБСОЛЮТНЫЕ ЗНАЧЕНИЯ:")
                print(f"      • Всего везикул: {total_vesicles}")
                print(f"      • Везикул в клетках (по маркеру): {vesicles_in_cells}")
                print(f"      • Колокализовано с белком: {colocalized_vesicles}")
                
                print(f"   📊 СРЕДНИЕ ЗНАЧЕНИЯ НА ОБРАЗЕЦ:")
                print(f"      • Везикул всего: {avg_total:.1f}")
                print(f"      • Везикул в клетках: {avg_in_cells:.1f}")
                print(f"      • Колокализовано с белком: {avg_colocalized:.1f}")
                
                print(f"   📊 ПРОЦЕНТЫ (от общего числа везикул):")
                print(f"      • % везикул в клетках: {in_cells_rate:.1f}%")
                print(f"      • % колокализации с белком: {colocalization_rate:.1f}%")
                
                print(f"   📊 СРЕДНИЕ ПРОЦЕНТЫ ПО ОБРАЗЦАМ:")
                print(f"      • Средний % везикул в клетках: {avg_percent_in_cells:.1f}%")
                print(f"      • Средний % колокализации: {avg_percentage:.1f}%")
            
            print(f"\n📁 РЕЗУЛЬТАТЫ СОХРАНЕНЫ В:")
            print(f"   • Основная папка: {output_dir}")
            print(f"   • Для каждого образца созданы папки по группам")
            
            print(f"\n💡 РЕКОМЕНДАЦИИ ДЛЯ АНАЛИЗА:")
            print(f"   1. Сравните файлы *_ORIGINAL.png и *_CORRECTED.png")
            print(f"   2. Проверьте аннотации в *_annotated.png файлах")
            print(f"   3. Результаты анализа в CSV файлах в папках образцов")
            print(f"   4. Сводная статистика в 00_analysis_summary.csv")
            
            print(f"\n⏱️  Время завершения: {datetime.now().strftime('%H:%M:%S')}")
            print("=" * 70)
            
        else:
            error_msg = result[1] if result else "❌ ОШИБКА: Неизвестная ошибка при обработке"
            print(f"\n{error_msg}")
            print(f"Проверьте файл LIF и пути доступа")
            
    except KeyboardInterrupt:
        print(f"\n⏹️  Анализ прерван пользователем")
        
    except Exception as e:
        error_msg = f"❌ ОШИБКА ПРИ ОБРАБОТКЕ: {str(e)}"
        print(f"\n{error_msg}")
        traceback.print_exc()
        
        print(f"\n🔧 ВОЗМОЖНЫЕ ПРИЧИНЫ:")
        print(f"   1. Файл LIF поврежден или имеет нестандартный формат")
        print(f"   2. Не хватает памяти для обработки")
        print(f"   3. Проблемы с правами доступа к файлам")
        print(f"   4. Ошибка в коде анализатора")

if __name__ == "__main__":
    main()
    print("\nНажмите Enter для выхода...")
    input()