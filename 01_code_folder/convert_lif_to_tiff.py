#!/usr/bin/env python3
"""
Конвертер LIF → многостраничный TIFF (uint16) с метаданными.
Переиспользует группировку из MultiSampleLifAnalyzer; загрузка каналов — без uint8.
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import tifffile
from PIL import Image
from readlif.reader import LifFile

from multi_lif_analyzer_legacy import MultiSampleLifAnalyzer

# === НАСТРОЙКИ ПО УМОЛЧАНИЮ (как в run_analysis.py) ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DEFAULT_LIF_FILE = os.path.join(
    PROJECT_ROOT,
    "02_raw_data",
    "lif_files",
    "FAPa 647 EVs 594 (1).lif",
)
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "preprocessed")

# Порядок каналов в LIF: ядра → коллаген → везикулы → белок-маркер
CHANNEL_ORDER = (0, 1, 2, 3)
CHANNEL_LABELS = {
    0: "nuclei",
    1: "collagen",
    2: "vesicles",
    3: "protein",
}
CHANNEL_LABELS_RU = {
    0: "ядра",
    1: "коллаген",
    2: "везикулы",
    3: "белок",
}


class LifToTiffConverter(MultiSampleLifAnalyzer):
    """
    Наследует метаданные/группировку из legacy.
    Загрузка каналов: load_all_channels_uint16() — аналог force_load_all_channels(),
    но без cv2.normalize и без деления uint16 на 256.
    """

    @staticmethod
    def to_uint16(array):
        """Приводит массив к uint16 без сжатия динамического диапазона в 8 бит."""
        data = np.asarray(array)
        if data.dtype == np.uint16:
            return data
        if data.dtype == np.uint8:
            return data.astype(np.uint16)
        if np.issubdtype(data.dtype, np.floating):
            if data.size and float(np.nanmax(data)) <= 1.0:
                scaled = data * 65535.0
            else:
                scaled = data
            return np.clip(scaled, 0, 65535).astype(np.uint16)
        return np.clip(data, 0, 65535).astype(np.uint16)

    def _frame_to_uint16(self, frame_data):
        if frame_data is None:
            return None
        if isinstance(frame_data, Image.Image):
            return self.to_uint16(self.pil_to_numpy(frame_data))
        return self.to_uint16(frame_data)

    def load_all_channels_uint16(self, image_dict, lif_file_path):
        """
        Загрузка всех каналов в исходном (16-битном) виде.
        Структура как у force_load_all_channels(), но без конвертации в uint8.
        """
        print("   🔍 Загрузка каналов (uint16, без нормализации)...")

        sample_channels = {}

        try:
            num_channels = image_dict.get("channels", 4)
            dims = image_dict.get("dims", None)
            image_name = image_dict.get("name", "unknown")

            print(f"   📐 Размеры: {dims}")
            print(f"   📊 Каналов: {num_channels}")
            print(f"   🏷️ Имя изображения: {image_name}")

            lif_file = LifFile(lif_file_path)

            image_index = None
            for i, img_item in enumerate(lif_file.image_list):
                img_name = (
                    img_item.get("name", "")
                    if isinstance(img_item, dict)
                    else getattr(img_item, "name", "")
                )
                if img_name == image_name:
                    image_index = i
                    break

            if image_index is None:
                print(f"   ❌ Не найден индекс изображения для: {image_name}")
                return {}

            print(f"   ✅ Найден индекс изображения: {image_index}")
            lif_image = lif_file.get_image(image_index)

            for channel_idx in range(num_channels):
                try:
                    print(f"   🔍 Загрузка канала {channel_idx}...")
                    channel_loaded = False

                    try:
                        channel_data = lif_image.get_frame(c=channel_idx)
                        channel_array = self._frame_to_uint16(channel_data)
                        if channel_array is not None:
                            sample_channels[channel_idx] = channel_array
                            channel_loaded = True
                            print(
                                f"   ✅ Канал {channel_idx}: "
                                f"{channel_array.dtype}, max={channel_array.max()}"
                            )
                    except Exception as exc:
                        print(f"   ❌ Ошибка загрузки канала {channel_idx} (метод 1): {exc}")

                    if not channel_loaded:
                        try:
                            all_channels = lif_image.get_frame()
                            if all_channels is None:
                                raise ValueError("get_frame() вернул None")

                            if isinstance(all_channels, Image.Image):
                                all_array = self.to_uint16(self.pil_to_numpy(all_channels))
                            else:
                                all_array = self.to_uint16(all_channels)

                            if len(all_array.shape) == 3 and all_array.shape[2] > channel_idx:
                                channel_array = all_array[:, :, channel_idx]
                                sample_channels[channel_idx] = channel_array
                                channel_loaded = True
                                print(f"   ✅ Канал {channel_idx} извлечён из 3D массива")
                            elif len(all_array.shape) == 2 and channel_idx == 0:
                                sample_channels[channel_idx] = all_array
                                channel_loaded = True
                                print(f"   ✅ Единственный канал (2D)")
                        except Exception as exc:
                            print(f"   ❌ Ошибка загрузки канала {channel_idx} (метод 2): {exc}")

                    if not channel_loaded:
                        print(f"   ❌ Не удалось загрузить канал {channel_idx}")

                except Exception as exc:
                    print(f"   ❌ Критическая ошибка канала {channel_idx}: {exc}")

            print(f"   📊 Успешно загружено каналов: {len(sample_channels)}")
            return sample_channels

        except Exception as exc:
            print(f"   ❌ Критическая ошибка загрузки каналов: {exc}")
            return {}

    def load_all_samples_from_lif_uint16(self, lif_file_path):
        """Как load_all_samples_from_lif(), но с uint16-каналами и без IgG-сканирования."""
        print(f"📖 Загрузка LIF (uint16): {lif_file_path}")

        try:
            lif_file = LifFile(lif_file_path)
            total_images = len(lif_file.image_list)
            print(f"✅ Найдено {total_images} изображений")
        except Exception as exc:
            print(f"❌ Ошибка загрузки LIF: {exc}")
            return {}

        samples_data = {}

        for i, image_item in enumerate(lif_file.image_list):
            print(f"\n🎯 Обработка изображения {i + 1}/{total_images}:")

            if isinstance(image_item, dict):
                image_name = image_item.get("name", f"image_{i + 1}")
            else:
                image_name = getattr(image_item, "name", f"image_{i + 1}")

            print(f"   Название: {image_name}")

            group_id = self.extract_sample_info(image_name)
            print(f"   Группа: {group_id}")

            sample_channels = self.load_all_channels_uint16(image_item, lif_file_path)
            if not sample_channels:
                print("   ⚠️ Не удалось загрузить каналы, пропускаем")
                continue

            samples_data.setdefault(group_id, []).append(
                {
                    "sample_name": image_name,
                    "channels": sample_channels,
                    "mouse_id": group_id,
                    "sample_slug": self.make_safe_name(image_name, fallback=f"sample_{i + 1}"),
                    "channels_count": len(sample_channels),
                    "original_index": i,
                    "is_igg": self.detect_igg_samples(image_name),
                }
            )
            print(f"   💾 Образец добавлен в группу: {group_id}")

        print("\n📊 СТАТИСТИКА ГРУППИРОВКИ:")
        total_samples = 0
        for group_id, samples in samples_data.items():
            total_samples += len(samples)
            igg_count = sum(1 for s in samples if s["is_igg"])
            print(f"   {group_id}: {len(samples)} образцов ({igg_count} контрольных)")

        print(f"   ВСЕГО: {total_samples} образцов")
        return samples_data

    def should_exclude_sample(self, sample_name):
        name_lower = str(sample_name or "").lower()
        return any(pattern in name_lower for pattern in self.exclude_patterns)

    @staticmethod
    def channel_statistics(channel_array):
        if channel_array is None:
            return None
        data = np.asarray(channel_array, dtype=np.float64)
        return {
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "shape": list(data.shape),
            "dtype": str(channel_array.dtype),
        }

    def build_channel_stack(self, sample_channels):
        """Собирает стек (4, H, W) uint16 в фиксированном порядке каналов."""
        missing = [idx for idx in CHANNEL_ORDER if idx not in sample_channels]
        if missing:
            raise ValueError(f"missing channels: {missing}")

        planes = []
        for idx in CHANNEL_ORDER:
            plane = self.to_uint16(sample_channels[idx])
            if plane.ndim != 2:
                raise ValueError(f"channel {idx} must be 2D, got shape {plane.shape}")
            planes.append(plane)

        stack = np.stack(planes, axis=0)
        if stack.dtype != np.uint16:
            stack = stack.astype(np.uint16)
        return stack

    def save_sample(self, output_root, group_id, sample_info, lif_file_path):
        sample_name = sample_info["sample_name"]
        sample_slug = sample_info.get("sample_slug") or self.make_safe_name(sample_name)
        sample_dir = os.path.join(output_root, group_id, sample_slug)
        os.makedirs(sample_dir, exist_ok=True)

        stack = self.build_channel_stack(sample_info["channels"])
        tiff_path = os.path.join(sample_dir, "channels.tif")
        tifffile.imwrite(
            tiff_path,
            stack,
            dtype=np.uint16,
            photometric="minisblack",
            metadata={"axes": "CYX"},
        )

        channels_meta = {}
        for idx in CHANNEL_ORDER:
            stats = self.channel_statistics(sample_info["channels"][idx])
            if stats is not None:
                stats["label"] = CHANNEL_LABELS[idx]
                stats["label_ru"] = CHANNEL_LABELS_RU[idx]
            channels_meta[str(idx)] = stats

        metadata = {
            "sample_name": sample_name,
            "sample_slug": sample_slug,
            "group": group_id,
            "original_index": sample_info.get("original_index"),
            "is_control": sample_info.get("is_igg", False),
            "channels_count": sample_info.get("channels_count"),
            "bit_depth": 16,
            "pixel_dtype": "uint16",
            "channel_order": list(CHANNEL_ORDER),
            "channel_labels": {str(k): v for k, v in CHANNEL_LABELS.items()},
            "channel_labels_ru": {str(k): v for k, v in CHANNEL_LABELS_RU.items()},
            "source_lif": os.path.abspath(lif_file_path),
            "tiff_file": os.path.basename(tiff_path),
            "converted_at": datetime.now().isoformat(timespec="seconds"),
            "channels": channels_meta,
        }

        metadata_path = os.path.join(sample_dir, "metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        return sample_dir, tiff_path, metadata_path

    def convert(self, lif_file_path, output_dir, skip_excluded=False):
        lif_file_path = os.path.abspath(lif_file_path)
        output_root = os.path.abspath(output_dir)
        os.makedirs(output_root, exist_ok=True)

        samples_by_group = self.load_all_samples_from_lif_uint16(lif_file_path)
        if not samples_by_group:
            print("❌ Образцы не загружены")
            return []

        saved = []
        skipped = []

        for group_id, samples in samples_by_group.items():
            for sample_info in samples:
                sample_name = sample_info["sample_name"]
                if skip_excluded and self.should_exclude_sample(sample_name):
                    skipped.append((group_id, sample_name, "exclude_patterns"))
                    print(f"⏭️  Пропуск (exclude_patterns): {sample_name}")
                    continue

                try:
                    paths = self.save_sample(output_root, group_id, sample_info, lif_file_path)
                    saved.append(paths)
                    print(f"✅ Сохранено: {paths[1]}")
                except Exception as exc:
                    print(f"❌ Ошибка сохранения {sample_name}: {exc}")

        print(f"\n📊 Итого: сохранено {len(saved)}, пропущено {len(skipped)}")
        return saved


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Конвертация LIF в многостраничный TIFF uint16 (каналы 0–3) с metadata.json",
        epilog=(
            "Без аргументов используются пути DEFAULT_LIF_FILE и DEFAULT_OUTPUT_DIR "
            "из начала файла (как run_analysis.py)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "lif_file",
        nargs="?",
        default=DEFAULT_LIF_FILE,
        help=f"Путь к .lif (по умолчанию: {DEFAULT_LIF_FILE})",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Папка вывода (по умолчанию: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--skip-excluded",
        action="store_true",
        help="Не сохранять образцы IgG/PBS/control (по exclude_patterns)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if not os.path.isfile(args.lif_file):
        print(f"❌ Файл не найден: {args.lif_file}")
        return 1

    converter = LifToTiffConverter(save_igg_data_path=None)
    converter.convert(
        args.lif_file,
        args.output,
        skip_excluded=args.skip_excluded,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
