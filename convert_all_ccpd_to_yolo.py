"""
CCPD 全数据集转换为 YOLO 格式脚本
支持 CCPD-2019, CCPD-2020, CCPD-BlueGreenYellow 三种数据集

功能：
1. 自动检测数据集类型
2. 支持三种 CCPD 文件名格式
3. 转换为 YOLO 格式（归一化坐标）
4. 创建 YOLOv5/YOLOv8 所需的目录结构
5. 自动复制图片和生成标签文件

CCPD 文件名格式：
- CCPD-2019/2020: 025-95_113-154&383_386&473-0&0_0&0_0&0-0-0.jpg
- CCPD-BlueGreenYellow: 0-0_0-0&342_719&610-714&610_0&585_15&342_719&367-29_16_2_2_30_31_28-0-0.jpg.png
"""

import os
import re
import shutil
import argparse
import random
from pathlib import Path
from tqdm import tqdm
import cv2
from typing import Tuple, Optional, List


def list_all_files(dirname, extensions=['.jpg', '.png', '.jpeg']):
    """
    遍历指定目录下的所有图片文件
    """
    result = []
    for root, _, files in os.walk(dirname):
        for filename in files:
            if any(filename.lower().endswith(ext) for ext in extensions):
                result.append(os.path.join(root, filename))
    return result


def extract_bbox_ccpd_standard(filename: str) -> Optional[Tuple[int, int, int, int]]:
    """
    从 CCPD 标准文件名中提取车牌边界框坐标
    适用于 CCPD-2019 和 CCPD-2020

    文件名格式示例：
    025-95_113-154&383_386&473-0&0_0&0_0&0-0-0.jpg
    0212703544062-86_91-244&503_460&597-465&574_259&593_258&514_464&495-0_0_16_25_25_29_13-140-72.jpg

    格式解析：
    - 第一部分：车牌号或角度
    - 第二部分：左上角坐标 (x0_y0)
    - 第三部分：右下角坐标 (x1&y1_x2&y2) - 取 x1&y1
    - 后续部分：四个顶点坐标、其他信息

    返回: (x0, y0, x1, y1) 或 None（如果解析失败）
    """
    try:
        # 移除文件扩展名
        basename = os.path.splitext(filename)[0]
        if basename.endswith('.jpg'):
            basename = os.path.splitext(basename)[0]

        # 提取包含坐标的部分：95_113-154&383
        pattern = r'-(\d+)_(\d+)-(\d+)&(\d+)_'
        match = re.search(pattern, basename)

        if match:
            x0 = int(match.group(1))
            y0 = int(match.group(2))
            x1 = int(match.group(3))
            y1 = int(match.group(4))
            return (x0, y0, x1, y1)
        else:
            return None
    except Exception as e:
        print(f"解析CCPD标准格式失败: {filename}, 错误: {e}")
        return None


def extract_bbox_ccpd_blueyellow(filename: str) -> Optional[Tuple[int, int, int, int]]:
    """
    从 CCPD-BlueGreenYellow 文件名中提取车牌边界框坐标

    文件名格式示例：
    0-0_0-0&342_719&610-714&610_0&585_15&342_719&367-29_16_2_2_30_31_28-0-0.jpg.png

    格式解析：
    - 第一部分：车牌号 (0)
    - 第二部分：左上角坐标 (0_0)
    - 第三部分：右下角坐标 (0&342_719&610) - 这里可能包含更多信息
    - 第四部分：另一个坐标点 (714&610)
    - 后续部分：四个顶点、其他信息

    实际上，这个格式更复杂。我们尝试从四个顶点坐标计算边界框。

    标准格式各部分含义：
    1. 车牌号
    2. 边界框左上角 (x0_y0)
    3. 边界框右下角 (x1&y1_x2&y2) - 四个角点的前两个
    4-7. 四个角点坐标
    8. 其他属性

    对于 BlueGreenYellow 格式，我们需要尝试多种模式。
    """
    try:
        # 移除文件扩展名（可能是 .jpg.png 或 .png）
        basename = filename
        if basename.endswith('.jpg.png'):
            basename = basename[:-8]
        elif basename.endswith('.png'):
            basename = basename[:-4]
        elif basename.endswith('.jpg'):
            basename = basename[:-4]

        # 模式1: 尝试使用标准格式（有些文件可能遵循标准）
        pattern1 = r'-(\d+)_(\d+)-(\d+)&(\d+)_'
        match = re.search(pattern1, basename)
        if match:
            x0 = int(match.group(1))
            y0 = int(match.group(2))
            x1 = int(match.group(3))
            y1 = int(match.group(4))
            # 验证坐标有效性
            if x0 >= 0 and y0 >= 0 and x1 > x0 and y1 > y0:
                return (x0, y0, x1, y1)

        # 模式2: BlueGreenYellow 特殊格式
        # 格式: 0-0_0-0&342_719&610-714&610_0&585_15&342_719&367-29_16_2_2_30_31_28-0-0
        # 我们需要提取四个顶点坐标来计算边界框

        # 尝试提取所有数字对
        # 按照标准 CCPD 格式，第4-7个字段应该是四个顶点坐标
        parts = basename.split('-')

        if len(parts) >= 8:
            # 第4-7部分应该是四个顶点 (格式: x0&y0_x1&y1_x2&y2_x3&y3)
            # 但实际上BlueGreenYellow格式可能不同

            # 尝试从第2和第3部分提取边界框
            # 第2部分: 0_0 (左上角)
            # 第3部分: 0&342_719&610 (这里格式不一样)

            # 让我们尝试从文件名中找到四个角点
            # 标准格式中：4个角点在第4-7部分

            if len(parts) >= 7:
                try:
                    # 尝试提取所有坐标点
                    all_coords = []
                    for part in parts[2:7]:  # 检查第3-7部分
                        if '&' in part:
                            # 可能包含坐标
                            subparts = part.split('_')
                            for sub in subparts:
                                if '&' in sub:
                                    coords = sub.split('&')
                                    if len(coords) >= 2:
                                        try:
                                            x, y = int(coords[0]), int(coords[1])
                                            if x >= 0 and y >= 0:
                                                all_coords.append((x, y))
                                        except:
                                            pass

                    # 如果找到了有效的坐标点，计算边界框
                    if len(all_coords) >= 2:
                        xs = [c[0] for c in all_coords]
                        ys = [c[1] for c in all_coords]
                        x0, x1 = min(xs), max(xs)
                        y0, y1 = min(ys), max(ys)

                        # 验证边界框
                        if x1 > x0 and y1 > y0:
                            return (x0, y0, x1, y1)
                except:
                    pass

        # 模式3: 使用更宽松的正则表达式
        # 尝试匹配任意两组坐标
        pattern3 = r'-?(\d+)_(\d+)-?(\d+)&(\d+)_?'
        match = re.search(pattern3, basename)
        if match:
            x0 = int(match.group(1))
            y0 = int(match.group(2))
            x1 = int(match.group(3))
            y1 = int(match.group(4))
            if x0 >= 0 and y0 >= 0 and x1 > x0 and y1 > y0:
                return (x0, y0, x1, y1)

        return None

    except Exception as e:
        print(f"解析BlueGreenYellow格式失败: {filename}, 错误: {e}")
        return None


def extract_bbox_from_filename(filename: str, dataset_type: str = 'auto') -> Optional[Tuple[int, int, int, int]]:
    """
    从 CCPD 文件名中提取车牌边界框坐标
    自动检测数据集类型

    参数:
        filename: 文件名
        dataset_type: 数据集类型 ('ccpd2019', 'ccpd2020', 'bluegreenyellow', 'auto')

    返回: (x0, y0, x1, y1) 或 None
    """
    if dataset_type == 'auto':
        # 自动检测：检查是否是 .jpg.png 结尾
        if filename.endswith('.jpg.png') or '.jpg.png' in filename:
            dataset_type = 'bluegreenyellow'
        else:
            dataset_type = 'ccpd2019'  # 默认使用标准格式

    if dataset_type == 'bluegreenyellow':
        return extract_bbox_ccpd_blueyellow(filename)
    else:
        return extract_bbox_ccpd_standard(filename)


def convert_to_yolo_format(bbox: Tuple[int, int, int, int], img_width: int, img_height: int) -> Tuple[int, float, float, float, float]:
    """
    将 CCPD 边界框坐标转换为 YOLO 格式

    参数:
        bbox: (x0, y0, x1, y1) CCPD 格式的边界框
        img_width: 图片宽度
        img_height: 图片高度

    返回: (class_id, x_center, y_center, width, height) 归一化到 0-1
    """
    x0, y0, x1, y1 = bbox

    # 计算中心点坐标和宽高
    x_center = (x0 + x1) / 2.0 / img_width
    y_center = (y0 + y1) / 2.0 / img_height
    width = (x1 - x0) / img_width
    height = (y1 - y0) / img_height

    # 限制坐标在 [0, 1] 范围内
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    width = max(0, min(1, width))
    height = max(0, min(1, height))

    # CCPD 数据集只有一个类别：车牌
    class_id = 0

    return (class_id, x_center, y_center, width, height)


def create_yolo_structure(base_path: str) -> List[str]:
    """
    创建 YOLO 数据集目录结构
    """
    directories = [
        os.path.join(base_path, 'images', 'train'),
        os.path.join(base_path, 'images', 'val'),
        os.path.join(base_path, 'images', 'test'),
        os.path.join(base_path, 'labels', 'train'),
        os.path.join(base_path, 'labels', 'val'),
        os.path.join(base_path, 'labels', 'test'),
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    print(f"✓ 已创建 YOLO 数据集目录结构: {base_path}")
    return directories


def detect_dataset_structure(source_path: str) -> dict:
    """
    自动检测数据集结构

    返回: {
        'type': 'ccpd2019' | 'ccpd2020' | 'bluegreenyellow',
        'has_splits': bool,
        'splits': dict  # train/val/test 的路径
    }
    """
    result = {
        'type': 'unknown',
        'has_splits': False,
        'splits': {}
    }

    # 检查是否存在 train/val/test 子目录
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(source_path, split)
        if os.path.exists(split_path):
            result['splits'][split] = split_path

    if len(result['splits']) > 0:
        result['has_splits'] = True

    # 检测数据集类型
    # 1. 通过路径名称检测
    path_lower = source_path.lower()
    if 'blue' in path_lower or 'green' in path_lower or 'yellow' in path_lower:
        result['type'] = 'bluegreenyellow'
    elif '2020' in path_lower:
        result['type'] = 'ccpd2020'
    elif '2019' in path_lower or 'ccpd' in path_lower:
        result['type'] = 'ccpd2019'

    # 2. 通过样本文件名检测
    sample_files = list_all_files(source_path)
    if len(sample_files) > 0:
        sample_filename = os.path.basename(sample_files[0])
        if sample_filename.endswith('.jpg.png') or '.jpg.png' in sample_filename:
            result['type'] = 'bluegreenyellow'
        elif 'ccpd' in path_lower and '2020' not in path_lower:
            result['type'] = 'ccpd2019'

    return result


def convert_dataset(
    source_path: str,
    target_path: str,
    dataset_type: str = 'auto',
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    copy_images: bool = True,
    preserve_splits: bool = False
) -> bool:
    """
    转换 CCPD 数据集为 YOLO 格式

    参数:
        source_path: CCPD 数据集源路径
        target_path: YOLO 数据集目标路径
        dataset_type: 数据集类型 ('auto', 'ccpd2019', 'ccpd2020', 'bluegreenyellow')
        val_ratio: 验证集比例（仅在 preserve_splits=False 时使用）
        test_ratio: 测试集比例（仅在 preserve_splits=False 时使用）
        copy_images: 是否复制图片（False 则移动图片）
        preserve_splits: 是否保留原始的 train/val/test 划分
    """
    print("\n" + "=" * 70)
    print("CCPD 数据集转换为 YOLO 格式")
    print("=" * 70)

    # 检查源路径
    if not os.path.exists(source_path):
        print(f"✗ 源路径不存在: {source_path}")
        return False

    # 检测数据集结构
    print("\n正在检测数据集结构...")
    dataset_info = detect_dataset_structure(source_path)

    if dataset_type == 'auto':
        dataset_type = dataset_info['type']

    print(f"✓ 数据集类型: {dataset_type}")
    print(f"✓ 包含预定义划分: {dataset_info['has_splits']}")

    if dataset_info['has_splits']:
        print(f"  检测到的划分: {list(dataset_info['splits'].keys())}")

    # 创建目标目录结构
    create_yolo_structure(target_path)

    total_processed = 0
    total_failed = 0

    if preserve_splits and dataset_info['has_splits']:
        # 使用原始的 train/val/test 划分
        print("\n使用原始数据集划分...")

        for split_name in ['train', 'val', 'test']:
            if split_name not in dataset_info['splits']:
                continue

            split_path = dataset_info['splits'][split_name]
            print(f"\n{'=' * 70}")
            print(f"处理 {split_name} 数据集...")
            print(f"{'=' * 70}")

            # 处理该划分
            processed, failed = process_split(
                source_path=split_path,
                target_path=target_path,
                split_name=split_name,
                dataset_type=dataset_type,
                copy_images=copy_images
            )

            total_processed += processed
            total_failed += failed

    else:
        # 自动划分数据集
        print("\n正在扫描所有图片文件...")
        image_files = list_all_files(source_path)

        if len(image_files) == 0:
            print(f"✗ 在 {source_path} 中未找到图片文件")
            return False

        print(f"✓ 找到 {len(image_files)} 个图片文件")

        # 划分数据集
        print("\n正在划分数据集...")
        random.seed(42)
        shuffled_files = image_files.copy()
        random.shuffle(shuffled_files)

        total_count = len(shuffled_files)
        test_count = int(total_count * test_ratio)
        val_count = int(total_count * val_ratio)
        train_count = total_count - test_count - val_count

        train_files = shuffled_files[:train_count]
        val_files = shuffled_files[train_count:train_count + val_count]
        test_files = shuffled_files[train_count + val_count:] if test_count > 0 else []

        print(f"✓ 训练集: {len(train_files)} 张")
        print(f"✓ 验证集: {len(val_files)} 张")
        if len(test_files) > 0:
            print(f"✓ 测试集: {len(test_files)} 张")

        # 处理每个划分
        for split_name, files in [('train', train_files), ('val', val_files)] + ([('test', test_files)] if test_files else []):
            print(f"\n{'=' * 70}")
            print(f"处理 {split_name} 数据集...")
            print(f"{'=' * 70}")

            processed, failed = process_files(
                files=files,
                target_path=target_path,
                split_name=split_name,
                dataset_type=dataset_type,
                copy_images=copy_images,
                start_idx=0
            )

            total_processed += processed
            total_failed += failed

    # 显示统计信息
    print(f"\n{'=' * 70}")
    print("转换完成!")
    print(f"{'=' * 70}")
    print(f"总计成功: {total_processed} 张")
    print(f"总计失败: {total_failed} 张")
    print(f"\nYOLO 数据集已保存到: {target_path}")

    return True


def process_split(
    source_path: str,
    target_path: str,
    split_name: str,
    dataset_type: str,
    copy_images: bool
) -> Tuple[int, int]:
    """处理单个数据划分"""
    image_files = list_all_files(source_path)

    if len(image_files) == 0:
        print(f"⚠ {split_name} 中没有找到图片文件")
        return 0, 0

    print(f"✓ 找到 {len(image_files)} 个图片文件")

    return process_files(
        files=image_files,
        target_path=target_path,
        split_name=split_name,
        dataset_type=dataset_type,
        copy_images=copy_images,
        start_idx=0
    )


def process_files(
    files: List[str],
    target_path: str,
    split_name: str,
    dataset_type: str,
    copy_images: bool,
    start_idx: int = 0
) -> Tuple[int, int]:
    """处理文件列表"""
    success = 0
    failed = 0

    for idx, img_path in enumerate(tqdm(files, desc=f"处理 {split_name}")):
        try:
            # 读取图片获取尺寸
            img = cv2.imread(img_path)
            if img is None:
                failed += 1
                continue

            height, width = img.shape[:2]

            # 从文件名提取边界框
            filename = os.path.basename(img_path)
            bbox = extract_bbox_from_filename(filename, dataset_type)

            if bbox is None:
                failed += 1
                continue

            # 转换为 YOLO 格式
            yolo_bbox = convert_to_yolo_format(bbox, width, height)

            # 生成新文件名
            file_ext = os.path.splitext(filename)[1]
            # 处理 .jpg.png 双重扩展名
            if file_ext == '.png' and filename.endswith('.jpg.png'):
                file_ext = '.png'
                base_name = filename[:-8]
            else:
                base_name = os.path.splitext(filename)[0]

            # 清理文件名中的特殊字符
            clean_name = re.sub(r'[^\w]', '_', base_name)
            new_filename = f"{split_name}_{str(start_idx + idx).zfill(6)}_{clean_name}{file_ext}"

            # 复制/移动图片
            img_target_path = os.path.join(target_path, 'images', split_name, new_filename)
            if copy_images:
                shutil.copy2(img_path, img_target_path)
            else:
                shutil.move(img_path, img_target_path)

            # 保存标签文件
            label_filename = os.path.splitext(new_filename)[0] + '.txt'
            label_target_path = os.path.join(target_path, 'labels', split_name, label_filename)

            with open(label_target_path, 'w') as f:
                # YOLO 格式: class_id x_center y_center width height
                f.write(f"{yolo_bbox[0]} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f} {yolo_bbox[4]:.6f}\n")

            success += 1

        except Exception as e:
            print(f"\n✗ 处理失败: {img_path}, 错误: {e}")
            failed += 1

    print(f"\n{split_name} 数据集处理完成:")
    print(f"  成功: {success}")
    print(f"  失败: {failed}")

    return success, failed


def create_yaml_config(target_path: str, dataset_name: str = 'CCPD') -> str:
    """
    创建 YOLO 数据集配置文件
    """
    yaml_content = f"""# CCPD 数据集配置文件
# 由 convert_all_ccpd_to_yolo.py 自动生成

# 数据集路径（相对于此文件）
path: {os.path.abspath(target_path)}  # 数据集根目录
train: images/train  # 训练集图片路径
val: images/val      # 验证集图片路径
test: images/test    # 测试集图片路径（可选）

# 类别数量
nc: 1

# 类别名称
names:
  0: license_plate
"""

    yaml_path = os.path.join(target_path, 'data.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)

    print(f"✓ 已创建数据集配置文件: {yaml_path}")
    return yaml_path


def main():
    parser = argparse.ArgumentParser(description='CCPD 全数据集转换为 YOLO 格式')
    parser.add_argument(
        '--source',
        type=str,
        default='./CCPD_Datasets',
        help='CCPD 数据集源路径（可以是包含多个数据集的目录，或单个数据集目录）'
    )
    parser.add_argument(
        '--target',
        type=str,
        default='./YOLO_Data',
        help='YOLO 数据集目标路径'
    )
    parser.add_argument(
        '--dataset-type',
        type=str,
        default='auto',
        choices=['auto', 'ccpd2019', 'ccpd2020', 'bluegreenyellow'],
        help='数据集类型（默认: auto 自动检测）'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.2,
        help='验证集比例（默认: 0.2，仅在 preserve-splits=False 时有效）'
    )
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.1,
        help='测试集比例（默认: 0.1，仅在 preserve-splits=False 时有效）'
    )
    parser.add_argument(
        '--copy',
        action='store_true',
        default=True,
        help='复制图片而非移动（保留原始文件，默认启用）'
    )
    parser.add_argument(
        '--no-yaml',
        action='store_true',
        help='不创建 data.yaml 配置文件'
    )
    parser.add_argument(
        '--preserve-splits',
        action='store_true',
        help='保留原始的 train/val/test 划分（如果存在）'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='转换所有 CCPD 数据集（CCPD2019, CCPD2020, CCPD-BlueGreenYellow）'
    )

    args = parser.parse_args()

    # 验证比例
    if args.val_ratio + args.test_ratio >= 1.0:
        print("✗ 验证集和测试集比例之和必须小于 1.0")
        return

    # 确定要转换的数据集
    datasets_to_convert = []

    if args.all:
        # 检测所有数据集
        base_path = args.source
        possible_datasets = [
            ('CCPD', 'CCPD/puhaiyang___CCPD2019/CCPD2019', 'ccpd2019'),
            ('CCPD2020', 'CCPD2020/puhaiyang___CCPD2020/CCPD2020', 'ccpd2020'),
            ('CCPD-BlueGreenYellow', 'CCPD_BlueGreenYellow/puhaiyang___ccpdblueyellowgreen/ccpd_blue_yellow_green', 'bluegreenyellow'),
        ]

        for name, relative_path, dtype in possible_datasets:
            full_path = os.path.join(base_path, relative_path)
            if os.path.exists(full_path):
                datasets_to_convert.append((name, full_path, dtype))
    else:
        # 转换单个数据集
        datasets_to_convert.append(('CCPD', args.source, args.dataset_type))

    if len(datasets_to_convert) == 0:
        print("✗ 未找到任何 CCPD 数据集")
        print("\n请检查路径或使用 --all 参数转换所有数据集")
        return

    print(f"找到 {len(datasets_to_convert)} 个数据集待转换")

    # 转换每个数据集
    for idx, (name, source_path, dtype) in enumerate(datasets_to_convert):
        if len(datasets_to_convert) > 1:
            print(f"\n\n{'#' * 70}")
            print(f"转换数据集 {idx + 1}/{len(datasets_to_convert)}: {name}")
            print(f"{'#' * 70}")

        # 设置目标路径
        if len(datasets_to_convert) > 1:
            target_path = os.path.join(args.target, name)
        else:
            target_path = args.target

        # 转换数据集
        success = convert_dataset(
            source_path=source_path,
            target_path=target_path,
            dataset_type=dtype,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            copy_images=args.copy,
            preserve_splits=args.preserve_splits
        )

        if success and not args.no_yaml:
            # 创建配置文件
            create_yaml_config(target_path, dataset_name=name)

    # 显示使用说明
    if len(datasets_to_convert) > 1:
        print(f"\n\n{'=' * 70}")
        print("所有数据集转换完成!")
        print(f"{'=' * 70}")
        print(f"数据集已保存到: {args.target}")
        print(f"\n各数据集配置文件:")
        for name, _, _ in datasets_to_convert:
            yaml_path = os.path.join(args.target, name, 'data.yaml')
            print(f"  - {name}: {yaml_path}")
    elif not args.no_yaml:
        print("\n" + "=" * 70)
        print("使用说明")
        print("=" * 70)
        print(f"1. YOLO 数据集已准备完成: {args.target}")
        print(f"2. 配置文件: {os.path.join(args.target, 'data.yaml')}")

        print(f"\n训练 YOLOv5 示例:")
        print(f"  python train.py --data {os.path.join(args.target, 'data.yaml')} --weights yolov5s.pt")

        print(f"\n训练 YOLOv8 示例:")
        print(f"  yolo detect train data={os.path.join(args.target, 'data.yaml')} model=yolov8n.pt")
        print("=" * 70)


if __name__ == '__main__':
    main()
