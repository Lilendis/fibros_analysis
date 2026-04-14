# fiji_converter.py
import os
import subprocess
import sys

def create_fiji_script(lif_file_path, output_dir):
    """Создает скрипт для Fiji для конвертации LIF в TIFF"""
    
    fiji_script = """
// Fiji Script для конвертации LIF в TIFF
// Сохранить как: convert_lif_to_tiff.ijm

// Параметры
lif_path = "%s";
output_dir = "%s";

// Открываем LIF файл
run("Bio-Formats", "open=[" + lif_path + "] autoscale color_mode=Default view=Hyperstack stack_order=XYCZT");
selectWindow(lif_path);

// Получаем информацию о изображении
getDimensions(width, height, channels, slices, frames);

print("LIF файл: " + lif_path);
print("Размеры: " + width + "x" + height);
print("Каналы: " + channels);
print("Срезы: " + slices);
print("Кадры: " + frames);

// Создаем выходную папку
File.makeDirectory(output_dir);

// Сохраняем каждый канал отдельно
for (c=1; c<=channels; c++) {
    // Выбираем канал
    Stack.setChannel(c);
    
    // Сохраняем как TIFF
    channel_name = "channel_" + c;
    output_path = output_dir + File.separator + channel_name + ".tif";
    saveAs("Tiff", output_path);
    print("Сохранен канал " + c + ": " + output_path);
}

// Закрываем изображение
close();

print("Конвертация завершена!");
""" % (lif_file_path.replace("\\", "\\\\"), output_dir.replace("\\", "\\\\"))
    
    script_path = os.path.join(output_dir, "convert_lif_to_tiff.ijm")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(fiji_script)
    
    return script_path

def run_fiji_conversion(fiji_path, script_path):
    """Запускает Fiji с созданным скриптом"""
    print(f"🚀 Запуск Fiji для конвертации...")
    
    if sys.platform == "win32":
        # Windows
        cmd = [fiji_path, "--headless", "-macro", script_path]
    else:
        # Linux/Mac
        cmd = [fiji_path, "--headless", "-macro", script_path]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print("✅ Конвертация успешно завершена!")
            return True
        else:
            print(f"❌ Ошибка конвертации: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Ошибка запуска Fiji: {e}")
        return False

def main():
    """Главная функция для конвертации LIF в TIFF"""
    print("============================================================")
    print("АВТОМАТИЧЕСКАЯ КОНВЕРТАЦИЯ LIF → TIFF ЧЕРЕЗ FIJI")
    print("============================================================")
    
    # Настройки - ИЗМЕНИТЕ ПУТИ ПОД ВАШУ СИСТЕМУ
    LIF_FILE_PATH = r"C:/Users/Petr/VSCode_Python/fibros_analysis/02_raw_data/lif_files/aSMa_647_EVs_594.lif"
    OUTPUT_BASE_DIR = r"C:/Users/Petr/VSCode_Python/fibros_analysis/02_raw_data/tiff_files"
    FIJI_PATH = r"C:/Users/Petr/Downloads/fiji-stable-win64-jdk/Fiji.app/fiji-windows-x64.exe"  # ИЗМЕНИТЕ НА ВАШ ПУТЬ К FIJI
    
    print(f"📁 LIF файл: {LIF_FILE_PATH}")
    print(f"📁 Выходная папка: {OUTPUT_BASE_DIR}")
    print(f"🔧 Fiji: {FIJI_PATH}")
    print("-" * 60)
    
    # Проверяем существование файлов
    if not os.path.exists(LIF_FILE_PATH):
        print(f"❌ LIF файл не найден: {LIF_FILE_PATH}")
        return
    
    if not os.path.exists(FIJI_PATH):
        print(f"❌ Fiji не найден по пути: {FIJI_PATH}")
        print("📋 Скачайте Fiji с https://fiji.sc/ и укажите правильный путь")
        return
    
    # Создаем папку для результатов
    base_name = os.path.splitext(os.path.basename(LIF_FILE_PATH))[0]
    output_dir = os.path.join(OUTPUT_BASE_DIR, base_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📂 Создана папка для TIFF файлов: {output_dir}")
    
    # Создаем скрипт для Fiji
    print("📝 Создание скрипта для Fiji...")
    script_path = create_fiji_script(LIF_FILE_PATH, output_dir)
    print(f"✅ Скрипт создан: {script_path}")
    
    # Запускаем конвертацию
    print("\n🔄 ЗАПУСК КОНВЕРТАЦИИ...")
    print("Это может занять несколько минут...")
    
    success = run_fiji_conversion(FIJI_PATH, script_path)
    
    if success:
        print(f"\n🎉 КОНВЕРТАЦИЯ УСПЕШНО ЗАВЕРШЕНА!")
        print(f"📁 TIFF файлы сохранены в: {output_dir}")
        print(f"🔍 Проверьте папку и запустите анализ TIFF файлов")
    else:
        print(f"\n❌ КОНВЕРТАЦИЯ НЕ УДАЛАСЯ")
        print("Попробуйте выполнить конвертацию вручную через Fiji")

if __name__ == "__main__":
    main()