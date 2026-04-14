# config.py
ANALYSIS_CONFIG = {
    # Размеры везикул (в пикселях)
    'min_vesicle_size': 5,
    'max_vesicle_size': 100,
    
    # Gamma коррекция для каналов
    'gamma_red': 0.7,      # Везикулы
    'gamma_blue': 0.8,     # Ядра
    'gamma_yellow': 0.8,   # Макрофаги
    
    # Параметры сегментации
    'adaptive_threshold': 11,
    'morphology_kernel': 3,
    
    # Настройки вывода
    'output_quality': 95,    # Качество PNG
    'dpi': 300              # Разрешение изображений
}

# Регулярные выражения для определения номеров мышей
MOUSE_PATTERNS = [
    r'mouse[_\s]*(\d+)', 
    r'm[_\s]*(\d+)',
    r'sample[_\s]*(\d+)', 
    r's[_\s]*(\d+)',
    r'(\d+)[^\d]*$'
]