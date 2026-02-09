#!/usr/bin/env python3
"""
改进的喇叭声检测器
使用相对阈值和自适应算法，更适合有背景噪音的场景
"""

import os
import numpy as np
import librosa
import soundfile as sf
from scipy.ndimage import binary_closing, binary_erosion, gaussian_filter1d
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class HornDetectorV2:
    """改进的喇叭声检测器 - 使用相对阈值"""

    def __init__(
        self,
        sample_rate: int = 22050,
        min_duration: float = 0.1,
        max_duration: float = 3.0,
        freq_range: Tuple[float, float] = (2000, 8000),
        relative_threshold: float = 2.0,  # 相对于背景噪音的倍数
        min_duration_frames: int = 5,  # 最小连续帧数
        merge_gap: float = 0.15,
    ):
        """
        初始化检测器

        Args:
            sample_rate: 音频采样率
            min_duration: 最短鸣笛时长（秒）
            max_duration: 最长鸣笛时长（秒）
            freq_range: 频率范围
            relative_threshold: 相对阈值（倍数，如 2.0 = 能量需要是背景的2倍）
            min_duration_frames: 最小连续帧数
            merge_gap: 合并间隔
        """
        self.sample_rate = sample_rate
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.freq_range = freq_range
        self.relative_threshold = relative_threshold
        self.min_duration_frames = min_duration_frames
        self.merge_gap = merge_gap

    def extract_audio(self, video_path: str, output_audio: str = None) -> str:
        """从视频提取音频"""
        print(f"[1/4] 从视频中提取音频...")

        if output_audio is None:
            output_audio = str(Path(video_path).with_suffix('.wav'))

        import subprocess
        cmd = [
            'ffmpeg', '-y', '-i', video_path,
            '-vn', '-acodec', 'pcm_s16le',
            '-ar', str(self.sample_rate), '-ac', '1',
            '-loglevel', 'error', output_audio
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"      ✅ 音频已提取")
            return output_audio
        except subprocess.CalledProcessError as e:
            print(f"      ❌ 音频提取失败: {e}")
            raise

    def detect_horns(self, audio_path: str) -> List[Dict]:
        """检测喇叭声 - 使用相对阈值"""
        print(f"[2/4] 分析音频检测喇叭声...")

        # 加载音频
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        duration = len(y) / sr

        # 计算频谱
        n_fft = 2048
        hop_length = 512

        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
        S_db = librosa.amplitude_to_db(S, ref=np.max)

        # 频率轴
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        times = librosa.times_like(S_db, sr=sr, hop_length=hop_length)

        # 目标频率范围
        freq_mask = (freqs >= self.freq_range[0]) & (freqs < self.freq_range[1])
        if not np.any(freq_mask):
            print(f"      ⚠️  频率范围无效")
            return []

        # 计算目标频带的能量轮廓
        energy_band = S_db[freq_mask, :]
        energy_profile = np.mean(energy_band, axis=0)

        # 平滑能量轮廓
        energy_smooth = gaussian_filter1d(energy_profile, sigma=2)

        # 计算背景噪音水平（使用滚动中位数）
        window_size = 100  # 约几秒
        background = np.convolve(
            energy_smooth,
            np.ones(window_size) / window_size,
            mode='same'
        )

        # 计算相对能量（相对于背景）
        relative_energy = energy_smooth - background

        print(f"      音频时长: {duration:.1f}秒")
        print(f"      频率范围: {self.freq_range[0]}-{self.freq_range[1]}Hz")
        print(f"      相对阈值: {self.relative_threshold}x 背景")
        print(f"      背景噪音: {np.median(background):.1f} dB")
        print(f"      峰值能量: {np.max(energy_smooth):.1f} dB")
        print(f"      最大相对能量: {np.max(relative_energy):.1f} dB")

        # 检测超过阈值的区域
        threshold_db = self.relative_threshold  # 使用dB作为阈值
        mask = relative_energy > threshold_db

        # 形态学操作：去除小的噪声点
        mask = binary_closing(mask, structure=np.ones(self.min_duration_frames))
        mask = binary_erosion(mask, structure=np.ones(2))

        # 找连续片段
        segments = []
        in_segment = False
        start_idx = 0

        for i, m in enumerate(mask):
            if m and not in_segment:
                in_segment = True
                start_idx = i
            elif not m and in_segment:
                in_segment = False
                end_idx = i
                segments.append((start_idx, end_idx))

        if in_segment:
            segments.append((start_idx, len(mask)))

        # 转换为时间并过滤
        horn_segments = []

        for start_idx, end_idx in segments:
            start_time = times[start_idx]
            end_time = times[end_idx]
            seg_duration = end_time - start_time

            # 时长过滤
            if seg_duration < self.min_duration or seg_duration > self.max_duration:
                continue

            # 计算置信度（基于相对能量）
            segment_relative_energy = relative_energy[start_idx:end_idx]
            confidence = float(np.max(segment_relative_energy) / self.relative_threshold)
            confidence = min(confidence, 1.0)  # 限制在 0-1

            horn_segments.append({
                'start': round(start_time, 2),
                'end': round(end_time, 2),
                'duration': round(seg_duration, 2),
                'confidence': round(confidence, 3),
                'max_relative_energy': round(float(np.max(segment_relative_energy)), 1)
            })

        # 合并相近片段
        horn_segments = self._merge_close_segments(horn_segments)
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
                current['end'] = next_seg['end']
                current['duration'] = current['end'] - current['start']
                current['confidence'] = max(current['confidence'], next_seg['confidence'])
            else:
                merged.append(current)
                current = next_seg

        merged.append(current)
        return merged

    def save_results(self, results: List[Dict], output_file: str):
        """保存检测结果"""
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
            print(f"\n{'序号':<6} {'开始时间':<12} {'结束时间':<12} {'时长(秒)':<10} {'置信度':<10} {'相对能量':<10}")
            print(f"{'-'*70}")

            for i, seg in enumerate(results, 1):
                print(f"{i:<6} {seg['start']:<12.2f} {seg['end']:<12.2f} {seg['duration']:<10.2f} "
                      f"{seg['confidence']:<10.3f} {seg.get('max_relative_energy', 0):<10.1f} dB")

        print(f"{'='*60}\n")

    def extract_audio_segments(
        self,
        audio_path: str,
        results: List[Dict],
        output_dir: str
    ):
        """提取音频片段"""
        if not results:
            print(f"[4/4] 没有检测到喇叭声")
            return

        print(f"[4/4] 提取喇叭声音频片段...")

        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        os.makedirs(output_dir, exist_ok=True)

        for i, seg in enumerate(results, 1):
            start_sample = int(seg['start'] * sr)
            end_sample = int(seg['end'] * sr)

            if end_sample > len(y):
                end_sample = len(y)

            segment_audio = y[start_sample:end_sample]
            output_file = os.path.join(output_dir, f"horn_{i:03d}_{seg['start']:.0f}s.wav")
            sf.write(output_file, segment_audio, sr)
            print(f"      ✅ {output_file}")

        print(f"      ✅ 共提取 {len(results)} 个片段")

    def process(
        self,
        video_path: str,
        output_dir: str = None,
        extract_segments: bool = True
    ):
        """完整处理流程"""
        print(f"{'='*60}")
        print(f"改进的喇叭声检测器")
        print(f"{'='*60}\n")

        if output_dir is None:
            video_name = Path(video_path).stem
            output_dir = os.path.join(os.path.dirname(video_path), f"{video_name}_horn_v2_output")

        os.makedirs(output_dir, exist_ok=True)

        # 1. 提取音频
        audio_path = os.path.join(output_dir, "extracted_audio.wav")
        audio_path = self.extract_audio(video_path, audio_path)

        # 2. 检测
        results = self.detect_horns(audio_path)

        # 3. 保存结果
        json_file = os.path.join(output_dir, "horn_detection.json")
        self.save_results(results, json_file)

        # 4. 打印摘要
        self.print_summary(results)

        # 5. 提取片段
        if extract_segments:
            segments_dir = os.path.join(output_dir, "audio_segments")
            self.extract_audio_segments(audio_path, results, segments_dir)

        print(f"✅ 处理完成！结果保存在: {output_dir}")

        return results


def main():
    parser = argparse.ArgumentParser(
        description='改进的喇叭声检测器（使用相对阈值）'
    )

    parser.add_argument('video', type=str, help='视频文件路径')
    parser.add_argument('--output-dir', type=str, help='输出目录')
    parser.add_argument('--freq-min', type=float, default=2000, help='最小频率')
    parser.add_argument('--freq-max', type=float, default=8000, help='最大频率')
    parser.add_argument('--threshold', type=float, default=3.0,
                        help='相对阈值（dB），越高越严格 (默认: 3.0)')
    parser.add_argument('--min-duration', type=float, default=0.1, help='最短时长')
    parser.add_argument('--max-duration', type=float, default=3.0, help='最长时长')
    parser.add_argument('--no-extract', action='store_true', help='不提取音频片段')

    args = parser.parse_args()

    # 检查文件
    if not os.path.exists(args.video):
        print(f"❌ 错误: 视频文件不存在: {args.video}")
        return

    # 创建检测器
    detector = HornDetectorV2(
        sample_rate=22050,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        freq_range=(args.freq_min, args.freq_max),
        relative_threshold=args.threshold,
    )

    # 处理
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
