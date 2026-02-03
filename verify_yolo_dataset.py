"""
YOLO 数据集验证脚本

功能：
1. 检查 YOLO 数据集完整性
2. 可视化标签（绘制边界框）
3. 统计数据集信息
4. 验证标签格式
"""

import os
import cv2
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm


class YOLODatasetVerifier:
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.images_train = self.dataset_path / 'images' / 'train'
        self.labels_train = self.dataset_path / 'labels' / 'train'
        self.images_val = self.dataset_path / 'images' / 'val'
        self.labels_val = self.dataset_path / 'labels' / 'val'

    def verify_structure(self):
        """验证目录结构"""
        print("\n" + "=" * 70)
        print("验证目录结构")
        print("=" * 70)

        required_dirs = [
            'images/train',
            'images/val',
            'labels/train',
            'labels/val'
        ]

        all_exist = True
        for dir_path in required_dirs:
            full_path = self.dataset_path / dir_path
            exists = full_path.exists()
            status = "✓" if exists else "✗"
            print(f"{status} {dir_path}: {'存在' if exists else '不存在'}")
            if not exists:
                all_exist = False

        return all_exist

    def count_files(self):
        """统计文件数量"""
        print("\n" + "=" * 70)
        print("统计文件数量")
        print("=" * 70)

        splits = {
            'train': (self.images_train, self.labels_train),
            'val': (self.images_val, self.labels_val)
        }

        total_images = 0
        total_labels = 0

        for split_name, (img_dir, lbl_dir) in splits.items():
            if img_dir.exists():
                img_count = len(list(img_dir.glob('*.*')))
                lbl_count = len(list(lbl_dir.glob('*.txt'))) if lbl_dir.exists() else 0

                print(f"\n{split_name.upper()}:")
                print(f"  图片: {img_count}")
                print(f"  标签: {lbl_count}")

                if img_count != lbl_count:
                    print(f"  ⚠ 警告: 图片和标签数量不匹配！")
                else:
                    print(f"  ✓ 匹配")

                total_images += img_count
                total_labels += lbl_count

        print(f"\n总计:")
        print(f"  图片: {total_images}")
        print(f"  标签: {total_labels}")

        return total_images, total_labels

    def verify_labels(self, split='train', sample_size=10):
        """验证标签格式"""
        print(f"\n" + "=" * 70)
        print(f"验证 {split} 标签格式（随机抽查 {sample_size} 个）")
        print("=" * 70)

        lbl_dir = self.dataset_path / 'labels' / split
        img_dir = self.dataset_path / 'images' / split

        if not lbl_dir.exists():
            print(f"✗ 标签目录不存在: {lbl_dir}")
            return False

        label_files = list(lbl_dir.glob('*.txt'))
        if len(label_files) == 0:
            print(f"✗ 没有找到标签文件")
            return False

        import random
        sample_files = random.sample(label_files, min(sample_size, len(label_files)))

        all_valid = True
        for lbl_file in sample_files:
            with open(lbl_file, 'r') as f:
                lines = f.readlines()

            valid = True
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    print(f"✗ {lbl_file.name}: 格式错误（应有 5 个值）")
                    valid = False
                    all_valid = False
                    continue

                try:
                    class_id = int(parts[0])
                    x, y, w, h = map(float, parts[1:5])

                    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                        print(f"✗ {lbl_file.name}: 坐标超出 [0,1] 范围")
                        valid = False
                        all_valid = False

                except ValueError:
                    print(f"✗ {lbl_file.name}: 数值格式错误")
                    valid = False
                    all_valid = False

            if valid:
                # 检查对应的图片文件是否存在
                img_file = img_dir / (lbl_file.stem + '.jpg')
                if not img_file.exists():
                    img_file = img_dir / (lbl_file.stem + '.png')
                if not img_file.exists():
                    print(f"⚠ {lbl_file.name}: 缺少对应的图片文件")
                else:
                    print(f"✓ {lbl_file.name}: 格式正确")

        return all_valid

    def visualize_sample(self, split='train', num_samples=4):
        """可视化样本（绘制边界框）"""
        print(f"\n" + "=" * 70)
        print(f"可视化 {split} 样本（显示 {num_samples} 个）")
        print("=" * 70)

        img_dir = self.dataset_path / 'images' / split
        lbl_dir = self.dataset_path / 'labels' / split

        if not img_dir.exists() or not lbl_dir.exists():
            print(f"✗ 目录不存在")
            return

        img_files = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))

        if len(img_files) == 0:
            print(f"✗ 没有找到图片文件")
            return

        import random
        sample_files = random.sample(img_files, min(num_samples, len(img_files)))

        fig, axes = plt.subplots(1, num_samples, figsize=(5*num_samples, 5))
        if num_samples == 1:
            axes = [axes]

        for idx, img_file in enumerate(sample_files):
            # 读取图片
            img = cv2.imread(str(img_file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 读取标签
            lbl_file = lbl_dir / (img_file.stem + '.txt')
            if lbl_file.exists():
                with open(lbl_file, 'r') as f:
                    labels = f.readlines()

                # 绘制边界框
                h, w = img.shape[:2]
                for label in labels:
                    parts = label.strip().split()
                    if len(parts) == 5:
                        class_id, x, y, bw, bh = map(float, parts)
                        class_id = int(class_id)

                        # 转换为像素坐标
                        x_center = x * w
                        y_center = y * h
                        box_w = bw * w
                        box_h = bh * h

                        # 计算左上角坐标
                        x0 = int(x_center - box_w / 2)
                        y0 = int(y_center - box_h / 2)

                        # 绘制矩形
                        rect = patches.Rectangle(
                            (x0, y0), box_w, box_h,
                            linewidth=2, edgecolor='red', facecolor='none'
                        )
                        axes[idx].add_patch(rect)

            axes[idx].imshow(img)
            axes[idx].axis('off')
            axes[idx].set_title(img_file.name)

        plt.tight_layout()
        save_path = self.dataset_path / f'{split}_samples.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ 可视化结果已保存: {save_path}")
        plt.close()

    def generate_report(self):
        """生成完整验证报告"""
        print("\n" + "=" * 70)
        print("YOLO 数据集验证报告")
        print("=" * 70)

        # 1. 验证结构
        structure_ok = self.verify_structure()

        # 2. 统计文件
        if structure_ok:
            total_imgs, total_lbls = self.count_files()

        # 3. 验证标签格式
        if structure_ok:
            train_ok = self.verify_labels('train', sample_size=5)
            val_ok = self.verify_labels('val', sample_size=5)

        # 4. 可视化样本
        if structure_ok:
            self.visualize_sample('train', num_samples=4)
            self.visualize_sample('val', num_samples=4)

        print("\n" + "=" * 70)
        print("验证完成！")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='验证 YOLO 数据集')
    parser.add_argument(
        '--path',
        type=str,
        default='./YOLO_Data',
        help='YOLO 数据集路径'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        choices=['train', 'val'],
        help='要验证的数据集划分'
    )
    parser.add_argument(
        '--visualize-only',
        action='store_true',
        help='仅执行可视化'
    )

    args = parser.parse_args()

    verifier = YOLODatasetVerifier(args.path)

    if args.visualize_only:
        verifier.visualize_sample(args.split, num_samples=8)
    else:
        verifier.generate_report()


if __name__ == '__main__':
    main()
