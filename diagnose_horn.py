#!/usr/bin/env python3
"""
电动车鸣笛声诊断工具
分析音频频谱特征，帮助调整检测参数
"""

import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib import rcParams
import json
from pathlib import Path

# 设置中文显示
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False


def extract_audio(video_path: str, output_wav: str = None):
    """从视频提取音频"""
    import subprocess

    if output_wav is None:
        output_wav = str(Path(video_path).with_suffix('.wav'))

    print(f"正在从视频提取音频...")

    cmd = [
        'ffmpeg', '-y', '-i', video_path,
        '-vn', '-acodec', 'pcm_s16le',
        '-ar', '22050', '-ac', '1',
        '-loglevel', 'error', output_wav
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"✅ 音频已提取: {output_wav}")
        return output_wav
    except subprocess.CalledProcessError as e:
        print(f"❌ 音频提取失败: {e}")
        return None


def analyze_audio_spectrum(audio_path: str):
    """分析音频频谱特征"""
    print(f"\n正在分析音频频谱...")

    # 加载音频
    y, sr = librosa.load(audio_path, sr=22050)
    duration = len(y) / sr

    print(f"音频时长: {duration:.1f} 秒")
    print(f"采样率: {sr} Hz")

    # 计算频谱
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    S_db = librosa.amplitude_to_db(S, ref=np.max)

    # 频率轴
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    times = librosa.times_like(S_db, sr=sr, hop_length=512)

    # 分析各个频段的能量分布
    freq_bands = {
        '低频 (0-1kHz)': (0, 1000),
        '中低频 (1-2kHz)': (1000, 2000),
        '中频 (2-4kHz)': (2000, 4000),
        '中高频 (4-6kHz)': (4000, 6000),
        '高频 (6-8kHz)': (6000, 8000),
        '超高频 (8-10kHz)': (8000, 10000),
    }

    print("\n各频段能量分析:")
    print("=" * 60)
    band_energies = {}

    for band_name, (f_min, f_max) in freq_bands.items():
        freq_mask = (freqs >= f_min) & (freqs < f_max)
        if np.any(freq_mask):
            band_energy = S_db[freq_mask, :]
            mean_energy = np.mean(band_energy)
            max_energy = np.max(band_energy)
            band_energies[band_name] = {
                'mean': float(mean_energy),
                'max': float(max_energy),
                'range': (f_min, f_max)
            }
            print(f"{band_name:20s} 平均: {mean_energy:6.2f} dB  最大: {max_energy:6.2f} dB")

    # 找到能量最高的频段
    max_band = max(band_energies.items(), key=lambda x: x[1]['max'])
    print(f"\n最高能量频段: {max_band[0]} ({max_band[1]['range'][0]}-{max_band[1]['range'][1]} Hz)")

    return y, sr, S_db, freqs, times, band_energies


def find_high_energy_segments(S_db, freqs, times, freq_min, freq_max, threshold_db=-30):
    """查找指定频段的高能量片段"""
    freq_mask = (freqs >= freq_min) & (freqs < freq_max)
    if not np.any(freq_mask):
        return []

    energy_profile = np.mean(S_db[freq_mask, :], axis=0)

    # 找到超过阈值的点
    above_threshold = energy_profile > threshold_db

    # 找连续片段
    segments = []
    in_segment = False
    start_idx = 0

    for i, above in enumerate(above_threshold):
        if above and not in_segment:
            in_segment = True
            start_idx = i
        elif not above and in_segment:
            in_segment = False
            end_idx = i
            segments.append((start_idx, end_idx))

    if in_segment:
        segments.append((start_idx, len(energy_profile)))

    # 转换为时间
    result = []
    for start_idx, end_idx in segments:
        start_time = times[start_idx]
        end_time = times[end_idx]
        duration = end_time - start_time
        if duration >= 0.1:  # 至少 0.1 秒
            segment_energy = energy_profile[start_idx:end_idx]
            result.append({
                'start': start_time,
                'end': end_time,
                'duration': duration,
                'max_energy': float(np.max(segment_energy)),
                'mean_energy': float(np.mean(segment_energy))
            })

    return sorted(result, key=lambda x: x['max_energy'], reverse=True)


def plot_spectrum_analysis(audio_path, output_dir):
    """生成频谱分析图"""
    print(f"\n正在生成频谱分析图...")

    y, sr = librosa.load(audio_path, sr=22050)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 图1：完整频谱图
    fig, ax = plt.subplots(figsize=(16, 6))

    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img = librosa.display.specshow(D, y_axis='hz', x_axis='time', sr=sr, ax=ax,
                                    fmax=10000, cmap='magma')
    ax.set_title('音频频谱图 (0-10kHz)', fontsize=14, fontweight='bold')
    ax.set_xlabel('时间 (秒)', fontsize=12)
    ax.set_ylabel('频率 (Hz)', fontsize=12)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'spectrum_full.png'), dpi=150, bbox_inches='tight')
    print(f"  ✅ 完整频谱图: spectrum_full.png")
    plt.close(fig)

    # 图2：高频段详细分析 (2-10kHz)
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

    freq_ranges = [(2000, 4000), (4000, 6000), (6000, 10000)]
    titles = ['2-4 kHz (汽车喇叭)', '4-6 kHz (电动车)', '6-10 kHz (高频电动车)']

    for idx, ((f_min, f_max), title) in enumerate(zip(freq_ranges, titles)):
        ax = axes[idx]

        # 计算频谱
        S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

        # 只显示目标频段
        freq_mask = (freqs >= f_min) & (freqs < f_max)
        S_band = S[freq_mask, :]
        S_band_db = librosa.amplitude_to_db(S_band, ref=np.max)

        times = librosa.times_like(S_band_db, sr=sr, hop_length=512)

        # 绘图
        im = ax.imshow(S_band_db, aspect='auto', origin='lower',
                       extent=[times[0], times[-1], f_min, f_max],
                       cmap='magma', vmin=-60, vmax=0)

        ax.set_ylabel(f'{title}\n频率 (Hz)', fontsize=11)
        ax.set_ylim(f_min, f_max)

        if idx == 2:
            ax.set_xlabel('时间 (秒)', fontsize=12)

        fig.colorbar(im, ax=ax, format='%+2.0f dB')

    fig.suptitle('高频段详细分析', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'spectrum_bands.png'), dpi=150, bbox_inches='tight')
    print(f"  ✅ 高频段分析图: spectrum_bands.png")
    plt.close(fig)

    # 图3：能量随时间变化
    fig, ax = plt.subplots(figsize=(16, 6))

    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    times = librosa.times_like(S, sr=sr, hop_length=512)

    # 各频带能量随时间
    freq_bands_colors = [
        ((0, 1000), 'blue', '0-1kHz'),
        ((1000, 2000), 'green', '1-2kHz'),
        ((2000, 4000), 'orange', '2-4kHz'),
        ((4000, 6000), 'red', '4-6kHz'),
        ((6000, 8000), 'purple', '6-8kHz'),
        ((8000, 10000), 'brown', '8-10kHz'),
    ]

    for (f_min, f_max), color, label in freq_bands_colors:
        freq_mask = (freqs >= f_min) & (freqs < f_max)
        energy = np.mean(S[freq_mask, :], axis=0)
        energy_db = librosa.amplitude_to_db(energy + 1e-9, ref=np.max)
        ax.plot(times, energy_db, label=label, color=color, linewidth=1.5, alpha=0.7)

    ax.set_xlabel('时间 (秒)', fontsize=12)
    ax.set_ylabel('能量 (dB)', fontsize=12)
    ax.set_title('各频带能量随时间变化', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-60, 0)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'energy_timeline.png'), dpi=150, bbox_inches='tight')
    print(f"  ✅ 能量时间线: energy_timeline.png")
    plt.close(fig)


def recommend_parameters(band_energies):
    """根据频谱分析推荐参数"""
    print(f"\n{'='*60}")
    print(f"参数推荐")
    print(f"{'='*60}\n")

    # 找到高频能量最高的频段
    high_freq_bands = {
        '2-4kHz': band_energies.get('中频 (2-4kHz)', {}),
        '4-6kHz': band_energies.get('中高频 (4-6kHz)', {}),
        '6-8kHz': band_energies.get('高频 (6-8kHz)', {}),
        '8-10kHz': band_energies.get('超高频 (8-10kHz)', {}),
    }

    # 按最大能量排序
    sorted_bands = sorted(high_freq_bands.items(), key=lambda x: x[1].get('max', -100), reverse=True)

    print("检测到高频能量分布:")
    for band_name, energy_info in sorted_bands:
        if energy_info:
            print(f"  {band_name}: 最大能量 {energy_info['max']:.1f} dB")

    # 推荐
    top_band = sorted_bands[0]
    if top_band[1]:
        freq_range = top_band[1]['range']
        print(f"\n推荐参数配置:")
        print(f"  频率范围: {freq_range[0]}-{freq_range[1]} Hz")
        print(f"  能量阈值: -35 dB (较宽松)")
        print(f"  谐波阈值: 0.2 (降低，电动车是纯音)")
        print(f"  最小时长: 0.1 秒")
        print(f"  最大时长: 2.0 秒")

        print(f"\n推荐命令:")
        print(f"python detect_horn.py video.mp4 \\")
        print(f"  --freq-min {freq_range[0]} \\")
        print(f"  --freq-max {freq_range[1]} \\")
        print(f"  --energy-threshold 0.3 \\")
        print(f"  --harmonic-threshold 0.2 \\")
        print(f"  --min-duration 0.1 \\")
        print(f"  --max-duration 2.0")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='诊断电动车鸣笛声')
    parser.add_argument('video', type=str, help='视频文件路径')
    parser.add_argument('--output-dir', type=str, default='horn_diagnosis', help='输出目录')

    args = parser.parse_args()

    print("="*60)
    print("电动车鸣笛声诊断工具")
    print("="*60)

    # 1. 提取音频
    audio_path = extract_audio(args.video)
    if not audio_path:
        return

    # 2. 分析频谱
    y, sr, S_db, freqs, times, band_energies = analyze_audio_spectrum(audio_path)

    # 3. 生成频谱图
    plot_spectrum_analysis(audio_path, args.output_dir)

    # 4. 推荐参数
    recommend_parameters(band_energies)

    print(f"\n{'='*60}")
    print(f"诊断完成！结果保存在: {args.output_dir}")
    print(f"{'='*60}")
    print(f"\n请查看生成的频谱图，找到鸣笛声对应的时间点，")
    print(f"然后调整 detect_horn.py 的参数重新检测。")


if __name__ == '__main__':
    main()
