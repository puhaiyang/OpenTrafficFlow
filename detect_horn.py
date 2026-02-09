#!/usr/bin/env python3
"""
视频喇叭声检测与提取工具
功能：
1. 从视频中提取音频
2. 检测喇叭声片段
3. 输出鸣笛时间点
4. 可选：提取喇叭声音频片段
"""

import os
import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from scipy.ndimage import binary_closing, binary_erosion
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class HornDetector:
    """喇叭声检测器"""

    def __init__(
        self,
        sample_rate: int = 22050,
        min_duration: float = 0.3,  # 最短鸣笛时长（秒）
        max_duration: float = 3.0,  # 最长鸣笛时长（秒）
        freq_range: Tuple[float, float] = (2000, 5000),  # 喇叭声频率范围
        energy_threshold: float = 0.6,  # 能量阈值（0-1）
        harmonic_threshold: float = 0.4,  # 谐波阈值（0-1）
        merge_gap: float = 0.2,  # 合并间隔小于该值的片段
    ):
        """
        初始化检测器

        Args:
            sample_rate: 音频采样率
            min_duration: 最短鸣笛时长（秒）
            max_duration: 最长鸣笛时长（秒）
            freq_range: 喇叭声频率范围
            energy_threshold: 能量阈值（相对值）
            harmonic_threshold: 谐波阈值
            merge_gap: 合并间隔
        """
        self.sample_rate = sample_rate
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.freq_range = freq_range
        self.energy_threshold = energy_threshold
        self.harmonic_threshold = harmonic_threshold
        self.merge_gap = merge_gap

    def extract_audio(self, video_path: str, output_audio: str = None) -> str:
        """
        从视频中提取音频

        Args:
            video_path: 视频文件路径
            output_audio: 输出音频文件路径（可选）

        Returns:
            提取的音频文件路径
        """
        print(f"[1/4] 从视频中提取音频...")

        if output_audio is None:
            output_audio = str(Path(video_path).with_suffix('.wav'))

        # 使用 ffmpeg 提取音频
        import subprocess
        cmd = [
            'ffmpeg', '-y', '-i', video_path,
            '-vn',  # 不处理视频
            '-acodec', 'pcm_s16le',  # PCM 16位编码
            '-ar', str(self.sample_rate),  # 采样率
            '-ac', '1',  # 单声道
            '-loglevel', 'error',  # 只显示错误
            output_audio
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"      ✅ 音频已提取: {output_audio}")
            return output_audio
        except subprocess.CalledProcessError as e:
            print(f"      ❌ 音频提取失败: {e}")
            raise

    def detect_horns(self, audio_path: str) -> List[Dict]:
        """
        检测音频中的喇叭声

        Args:
            audio_path: 音频文件路径

        Returns:
            检测到的喇叭声片段列表
            [{'start': 开始时间, 'end': 结束时间, 'confidence': 置信度}, ...]
        """
        print(f"[2/4] 分析音频检测喇叭声...")

        # 加载音频
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        duration = len(y) / sr

        # 计算短时傅里叶变换
        n_fft = 2048
        hop_length = 512

        # 使用更短的窗口以获得更好的时间分辨率
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
        S_db = librosa.amplitude_to_db(S, ref=np.max)

        # 频率轴
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

        # 找到喇叭声频率范围
        freq_mask = (freqs >= self.freq_range[0]) & (freqs <= self.freq_range[1])
        freq_indices = np.where(freq_mask)[0]

        if len(freq_indices) == 0:
            print("      ⚠️  未找到目标频率范围")
            return []

        # 计算频域能量
        energy_band = S_db[freq_indices, :]
        energy_profile = np.mean(energy_band, axis=0)

        # 归一化
        energy_max = np.max(energy_profile)
        if energy_max > 0:
            energy_profile = energy_profile / energy_max

        # 时间轴
        times = librosa.times_like(S_db, sr=sr, hop_length=hop_length)

        print(f"      音频时长: {duration:.1f}秒")
        print(f"      分析帧数: {len(times)}")
        print(f"      频率范围: {self.freq_range[0]}-{self.freq_range[1]}Hz")

        # 能量阈值检测
        energy_mask = energy_profile > self.energy_threshold

        # 形态学操作：去除小段噪声，连接相近片段
        structure = np.ones(3)
        energy_mask = binary_closing(energy_mask, structure=structure)
        energy_mask = binary_erosion(energy_mask, structure=structure)

        # 找到连续的片段
        segments = []
        in_segment = False
        start_idx = 0

        for i, mask in enumerate(energy_mask):
            if mask and not in_segment:
                in_segment = True
                start_idx = i
            elif not mask and in_segment:
                in_segment = False
                end_idx = i
                segments.append((start_idx, end_idx))

        if in_segment:
            segments.append((start_idx, len(energy_mask)))

        # 转换为时间戳并过滤
        horn_segments = []

        for start_idx, end_idx in segments:
            start_time = times[start_idx]
            end_time = times[end_idx]
            seg_duration = end_time - start_time

            # 时长过滤
            if seg_duration < self.min_duration or seg_duration > self.max_duration:
                continue

            # 计算该片段的置信度
            segment_energy = energy_profile[start_idx:end_idx]
            confidence = float(np.mean(segment_energy))

            # 谐波检测：计算频谱平坦度
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            if end_sample > len(y):
                end_sample = len(y)

            segment_audio = y[start_sample:end_sample]

            # 计算频谱平坦度（喇叭声应该有明显的谐波结构，平坦度较低）
            if len(segment_audio) > n_fft:
                S_seg = np.abs(librosa.stft(segment_audio, n_fft=n_fft))
                flatness = librosa.feature.spectral_flatness(y=segment_audio)[0]
                harmonic_score = 1.0 - np.mean(flatness)

                # 结合能量和谐波特征
                final_confidence = (confidence + harmonic_score) / 2
            else:
                final_confidence = confidence

            # 谐波阈值过滤
            if final_confidence >= self.harmonic_threshold:
                horn_segments.append({
                    'start': round(start_time, 2),
                    'end': round(end_time, 2),
                    'duration': round(seg_duration, 2),
                    'confidence': round(final_confidence, 3)
                })

        # 合并间隔很小的片段
        horn_segments = self._merge_close_segments(horn_segments)

        # 按开始时间排序
        horn_segments.sort(key=lambda x: x['start'])

        return horn_segments

    def _merge_close_segments(self, segments: List[Dict]) -> List[Dict]:
        """合并间隔很小的片段"""
        if len(segments) <= 1:
            return segments

        merged = []
        current = segments[0]

        for next_seg in segments[1:]:
            if next_seg['start'] - current['end'] <= self.merge_gap:
                # 合并
                current['end'] = next_seg['end']
                current['duration'] = current['end'] - current['start']
                current['confidence'] = max(current['confidence'], next_seg['confidence'])
            else:
                merged.append(current)
                current = next_seg

        merged.append(current)
        return merged

    def save_results(self, results: List[Dict], output_file: str):
        """保存检测结果到 JSON 文件"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'total_count': len(results),
                'segments': results
            }, f, ensure_ascii=False, indent=2)
        print(f"[3/4] 检测结果已保存: {output_file}")

    def print_summary(self, results: List[Dict]):
        """打印检测摘要"""
        print(f"\n{'='*60}")
        print(f"喇叭声检测摘要")
        print(f"{'='*60}")
        print(f"检测到鸣笛次数: {len(results)} 次")

        if results:
            print(f"\n{'序号':<6} {'开始时间':<12} {'结束时间':<12} {'时长(秒)':<10} {'置信度':<10}")
            print(f"{'-'*60}")

            for i, seg in enumerate(results, 1):
                print(f"{i:<6} {seg['start']:<12.2f} {seg['end']:<12.2f} {seg['duration']:<10.2f} {seg['confidence']:<10.3f}")

        print(f"{'='*60}\n")

    def extract_audio_segments(
        self,
        audio_path: str,
        results: List[Dict],
        output_dir: str
    ):
        """
        提取喇叭声音频片段

        Args:
            audio_path: 原始音频文件
            results: 检测结果
            output_dir: 输出目录
        """
        if not results:
            print(f"[4/4] 没有检测到喇叭声，跳过音频提取")
            return

        print(f"[4/4] 提取喇叭声音频片段...")

        # 加载音频
        y, sr = librosa.load(audio_path, sr=self.sample_rate)

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        extracted_files = []

        for i, seg in enumerate(results, 1):
            start_sample = int(seg['start'] * sr)
            end_sample = int(seg['end'] * sr)

            # 边界检查
            if end_sample > len(y):
                end_sample = len(y)

            segment_audio = y[start_sample:end_sample]

            # 保存片段
            output_file = os.path.join(output_dir, f"horn_{i:03d}.wav")
            sf.write(output_file, segment_audio, sr)
            extracted_files.append(output_file)

            print(f"      ✅ 提取片段 {i}/{len(results)}: {output_file}")

        print(f"      ✅ 共提取 {len(extracted_files)} 个音频片段到: {output_dir}")

    def process(
        self,
        video_path: str,
        output_dir: str = None,
        extract_segments: bool = True
    ):
        """
        完整处理流程

        Args:
            video_path: 视频文件路径
            output_dir: 输出目录
            extract_segments: 是否提取音频片段
        """
        print(f"{'='*60}")
        print(f"开始处理视频: {os.path.basename(video_path)}")
        print(f"{'='*60}\n")

        # 设置输出目录
        if output_dir is None:
            video_name = Path(video_path).stem
            output_dir = os.path.join(os.path.dirname(video_path), f"{video_name}_horn_output")

        os.makedirs(output_dir, exist_ok=True)

        # 1. 提取音频
        audio_path = os.path.join(output_dir, "extracted_audio.wav")
        audio_path = self.extract_audio(video_path, audio_path)

        # 2. 检测喇叭声
        results = self.detect_horns(audio_path)

        # 3. 保存结果
        json_file = os.path.join(output_dir, "horn_detection.json")
        self.save_results(results, json_file)

        # 4. 打印摘要
        self.print_summary(results)

        # 5. 提取音频片段
        if extract_segments:
            segments_dir = os.path.join(output_dir, "audio_segments")
            self.extract_audio_segments(audio_path, results, segments_dir)

        print(f"✅ 处理完成！结果保存在: {output_dir}")

        return results


def main():
    # 预设配置
    PRESETS = {
        'car': {
            'name': '汽车喇叭',
            'freq_min': 2000,
            'freq_max': 5000,
            'min_duration': 0.3,
            'max_duration': 3.0,
            'energy_threshold': 0.6,
            'harmonic_threshold': 0.4,
        },
        'ebike': {
            'name': '电动车/摩托车',
            'freq_min': 3500,
            'freq_max': 8000,
            'min_duration': 0.1,
            'max_duration': 2.0,
            'energy_threshold': 0.35,
            'harmonic_threshold': 0.2,
        },
        'sensitive': {
            'name': '高灵敏度模式',
            'freq_min': 1500,
            'freq_max': 10000,
            'min_duration': 0.1,
            'max_duration': 5.0,
            'energy_threshold': 0.25,
            'harmonic_threshold': 0.15,
        },
    }

    parser = argparse.ArgumentParser(
        description='检测视频中的喇叭声并提取时间点',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本使用（汽车喇叭）
  python detect_horn.py video.mp4

  # 电动车鸣笛检测
  python detect_horn.py video.mp4 --mode ebike

  # 高灵敏度检测（可能误报）
  python detect_horn.py video.mp4 --mode sensitive

  # 自定义参数
  python detect_horn.py video.mp4 --freq-min 1500 --freq-max 6000 --energy-threshold 0.5

  # 只检测不提取音频片段
  python detect_horn.py video.mp4 --no-extract

  # 指定输出目录
  python detect_horn.py video.mp4 --output-dir ./results

预设模式:
  car       - 汽车喇叭 (2-5kHz, 有明显谐波)
  ebike     - 电动车/摩托车 (3.5-8kHz, 纯音)
  sensitive - 高灵敏度，检测所有高频声音
        """
    )

    parser.add_argument('video', type=str, help='输入视频文件路径')
    parser.add_argument('--mode', type=str, choices=['car', 'ebike', 'sensitive'], default='car',
                        help='检测模式: car(汽车), ebike(电动车), sensitive(高灵敏度) (默认: car)')
    parser.add_argument('--output-dir', type=str, help='输出目录')
    parser.add_argument('--sample-rate', type=int, default=22050, help='音频采样率 (默认: 22050)')
    parser.add_argument('--min-duration', type=float, help='最短鸣笛时长(秒)')
    parser.add_argument('--max-duration', type=float, help='最长鸣笛时长(秒)')
    parser.add_argument('--freq-min', type=float, help='最小频率(Hz)')
    parser.add_argument('--freq-max', type=float, help='最大频率(Hz)')
    parser.add_argument('--energy-threshold', type=float, help='能量阈值 0-1')
    parser.add_argument('--harmonic-threshold', type=float, help='谐波阈值 0-1')
    parser.add_argument('--no-extract', action='store_true', help='不提取音频片段')

    args = parser.parse_args()

    # 应用预设配置
    preset = PRESETS[args.mode]
    freq_min = args.freq_min if args.freq_min is not None else preset['freq_min']
    freq_max = args.freq_max if args.freq_max is not None else preset['freq_max']
    min_duration = args.min_duration if args.min_duration is not None else preset['min_duration']
    max_duration = args.max_duration if args.max_duration is not None else preset['max_duration']
    energy_threshold = args.energy_threshold if args.energy_threshold is not None else preset['energy_threshold']
    harmonic_threshold = args.harmonic_threshold if args.harmonic_threshold is not None else preset['harmonic_threshold']

    print(f"使用预设模式: {preset['name']}")
    print(f"参数: 频率 {freq_min}-{freq_max}Hz, 能量阈值 {energy_threshold}, 谐波阈值 {harmonic_threshold}")

    # 检查视频文件
    if not os.path.exists(args.video):
        print(f"❌ 错误: 视频文件不存在: {args.video}")
        return

    # 创建检测器
    detector = HornDetector(
        sample_rate=args.sample_rate,
        min_duration=min_duration,
        max_duration=max_duration,
        freq_range=(freq_min, freq_max),
        energy_threshold=energy_threshold,
        harmonic_threshold=harmonic_threshold,
    )

    # 处理视频
    try:
        detector.process(
            video_path=args.video,
            output_dir=args.output_dir,
            extract_segments=not args.no_extract
        )
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
