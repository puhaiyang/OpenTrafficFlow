"""
CCPD 数据集转换为 YOLO 格式脚本

功能：
1. 从 CCPD 文件名中提取车牌位置信息
2. 转换为 YOLO 格式（归一化坐标）
3. 创建 YOLOv5 所需的目录结构
4. 支持 train/val/test 数据集划分
5. 自动复制图片和生成标签文件

CCPD 文件名格式：
025-95_113-154&383_386&473-0&0_0&0_0&0-0-0.jpg
  ├── 车牌倾斜角度: 025
  ├── 车牌边界框左上角坐标: 95_113
  ├── 车牌边界框右下角坐标: 154&383_386&473
  └── 其他信息...
"""

import os
import re
import shutil
import argparse
import random
from pathlib import Path
from tqdm import tqdm
import cv2


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


def extract_bbox_from_filename(filename):
    """
    从 CCPD 文件名中提取车牌边界框坐标

    文件名格式示例：
    025-95_113-154&383_386&473-0&0_0&0_0&0-0-0.jpg

    返回: (x0, y0, x1, y1) 或 None（如果解析失败）
    """
    try:
        # 提取边界框坐标：95_113-154&383_386&473
        # 方法1：提取第二部分（左上角）和第三部分（右侧两个点）
        # 使用正则提取坐标部分

        # 提取左上角坐标 (95_113)
        tl_pattern = r'-(\d+)_(\d+)-'
        tl_match = re.search(tl_pattern, filename)
        if not tl_match:
            return None
        x0 = int(tl_match.group(1))
        y0 = int(tl_match.group(2))

        # 提取右侧两个点 (154&383_386&473)，取第二个点作为右下角
        str1 = re.findall(r'-\d+&\d+_\d+&\d+-', filename)
        if not str1:
            return None

        # 去掉首尾的 '-'，然后按 '&' 或 '_' 分割
        str2 = re.split(r'&|_', str1[0][1:-1])

        # str2 = ['154', '383', '386', '473']
        # 取第二个点 (386, 473) 作为右下角
        x1 = int(str2[2])
        y1 = int(str2[3])

        return (x0, y0, x1, y1)
    except Exception as e:
        print(f"解析文件名失败: {filename}, 错误: {e}")
        return None


def convert_to_yolo_format(bbox, img_width, img_height):
    """
    将 CCPD 边界框坐标转换为 YOLO 格式（与 ConvertYOLOFormat.py 格式一致）

    参数:
        bbox: (x0, y0, x1, y1) CCPD 格式的边界框
        img_width: 图片宽度
        img_height: 图片高度

    返回: (class_id, x_center, y_center, width, height) 归一化到 0-1
    """
    x0, y0, x1, y1 = bbox

    # 计算中心点坐标和宽高（与 ConvertYOLOFormat.py 完全一致）
    x_center = round((x0 + x1) / 2 / img_width, 6)
    y_center = round((y0 + y1) / 2 / img_height, 6)
    width = round((x1 - x0) / img_width, 6)
    height = round((y1 - y0) / img_height, 6)

    # CCPD 数据集只有一个类别：车牌
    class_id = 0

    return (class_id, x_center, y_center, width, height)


def create_yolo_structure(base_path):
    """
    创建 YOLOv5 数据集目录结构
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


def convert_dataset(source_path, target_path, val_ratio=0.2, test_ratio=0.1, copy_images=True):
    """
    转换 CCPD 数据集为 YOLO 格式

    参数:
        source_path: CCPD 数据集源路径
        target_path: YOLO 数据集目标路径
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        copy_images: 是否复制图片（False 则移动图片）
    """
    print("\n" + "=" * 70)
    print("CCPD 数据集转换为 YOLO 格式")
    print("=" * 70)

    # 检查源路径
    if not os.path.exists(source_path):
        print(f"✗ 源路径不存在: {source_path}")
        return False

    # 创建目标目录结构
    create_yolo_structure(target_path)

    # 获取所有图片文件
    print("\n正在扫描图片文件...")
    image_files = list_all_files(source_path)

    if len(image_files) == 0:
        print(f"✗ 在 {source_path} 中未找到图片文件")
        return False

    print(f"✓ 找到 {len(image_files)} 个图片文件")

    # 划分数据集
    print("\n正在划分数据集...")
    # 设置随机种子以保证可重复性
    random.seed(42)

    # 打乱文件列表
    shuffled_files = image_files.copy()
    random.shuffle(shuffled_files)

    total_count = len(shuffled_files)
    test_count = int(total_count * test_ratio)
    val_count = int(total_count * val_ratio)
    train_count = total_count - test_count - val_count

    # 划分数据集
    train_files = shuffled_files[:train_count]
    val_files = shuffled_files[train_count:train_count + val_count]
    test_files = shuffled_files[train_count + val_count:] if test_count > 0 else []

    print(f"✓ 训练集: {len(train_files)} 张")
    print(f"✓ 验证集: {len(val_files)} 张")
    if len(test_files) > 0:
        print(f"✓ 测试集: {len(test_files)} 张")

    # 处理每个数据集
    splits = [
        ('train', train_files),
        ('val', val_files),
    ]

    if len(test_files) > 0:
        splits.append(('test', test_files))

    total_processed = 0
    total_failed = 0

    for split_name, files in splits:
        print(f"\n{'=' * 70}")
        print(f"处理 {split_name} 数据集...")
        print(f"{'=' * 70}")

        split_success = 0
        split_failed = 0

        for idx, img_path in enumerate(tqdm(files, desc=f"处理 {split_name}")):
            try:
                # 读取图片获取尺寸
                img = cv2.imread(img_path)
                if img is None:
                    print(f"\n⚠ 无法读取图片: {img_path}")
                    split_failed += 1
                    continue

                height, width = img.shape[:2]

                # 从文件名提取边界框
                filename = os.path.basename(img_path)
                bbox = extract_bbox_from_filename(filename)

                if bbox is None:
                    print(f"\n⚠ 无法解析文件名: {filename}")
                    split_failed += 1
                    continue

                # 转换为 YOLO 格式
                yolo_bbox = convert_to_yolo_format(bbox, width, height)

                # 生成新文件名
                file_ext = os.path.splitext(filename)[1]
                new_filename = f"ccpd_{split_name}_{str(idx).zfill(6)}{file_ext}"

                # 复制/移动图片
                img_target_path = os.path.join(target_path, 'images', split_name, new_filename)
                if copy_images:
                    shutil.copy2(img_path, img_target_path)
                else:
                    shutil.move(img_path, img_target_path)

                # 保存标签文件
                label_filename = new_filename.replace(file_ext, '.txt')
                label_target_path = os.path.join(target_path, 'labels', split_name, label_filename)

                with open(label_target_path, 'w') as f:
                    # YOLO 格式: class_id x_center y_center width height（与 ConvertYOLOFormat.py 格式一致）
                    f.write(" ".join([str(yolo_bbox[0]), str(yolo_bbox[1]), str(yolo_bbox[2]), str(yolo_bbox[3]), str(yolo_bbox[4])]) + "\n")

                split_success += 1

            except Exception as e:
                print(f"\n✗ 处理失败: {img_path}, 错误: {e}")
                split_failed += 1

        print(f"\n{split_name} 数据集处理完成:")
        print(f"  成功: {split_success}")
        print(f"  失败: {split_failed}")

        total_processed += split_success
        total_failed += split_failed

    # 显示统计信息
    print(f"\n{'=' * 70}")
    print("转换完成!")
    print(f"{'=' * 70}")
    print(f"总计成功: {total_processed} 张")
    print(f"总计失败: {total_failed} 张")
    print(f"\nYOLO 数据集已保存到: {target_path}")

    return True


def create_yaml_config(target_path, dataset_name='CCPD'):
    """
    创建 YOLOv5 数据集配置文件
    """
    yaml_content = f"""# CCPD 数据集配置文件
# 由 convert_ccpd_to_yolo.py 自动生成

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
    parser = argparse.ArgumentParser(description='CCPD 数据集转换为 YOLO 格式')
    parser.add_argument(
        '--source',
        type=str,
        default='./CCPD_Datasets/CCPD2020',
        help='CCPD 数据集源路径（包含 train/test 等文件夹）'
    )
    parser.add_argument(
        '--target',
        type=str,
        default='./YOLO_Data',
        help='YOLO 数据集目标路径'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.2,
        help='验证集比例（默认: 0.2）'
    )
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.1,
        help='测试集比例（默认: 0.1）'
    )
    parser.add_argument(
        '--copy',
        action='store_true',
        help='复制图片而非移动（保留原始文件）'
    )
    parser.add_argument(
        '--no-yaml',
        action='store_true',
        help='不创建 data.yaml 配置文件'
    )

    args = parser.parse_args()

    # 验证比例
    if args.val_ratio + args.test_ratio >= 1.0:
        print("✗ 验证集和测试集比例之和必须小于 1.0")
        return

    # 转换数据集
    success = convert_dataset(
        source_path=args.source,
        target_path=args.target,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        copy_images=args.copy
    )

    if success and not args.no_yaml:
        # 创建配置文件
        create_yaml_config(args.target)

        print("\n" + "=" * 70)
        print("使用说明")
        print("=" * 70)
        print(f"1. YOLO 数据集已准备完成: {args.target}")
        print(f"2. 配置文件: {os.path.join(args.target, 'data.yaml')}")
        print(f"\n训练 YOLOv5 示例:")
        print(f"  python train.py --data {os.path.join(args.target, 'data.yaml')} --weights yolov5s.pt")
        print("=" * 70)


if __name__ == '__main__':
    main()
