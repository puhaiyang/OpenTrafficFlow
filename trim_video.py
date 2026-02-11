#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
视频剪辑工具 - 保留GPS信息到JSON

实现方式：
1. 使用 FFmpeg 剪辑视频片段
2. 使用 pyosmogps 提取 GPS 数据
3. 将 GPS 信息保存为 JSON 文件
4. 输出剪辑后的视频和 GPS JSON 文件
"""
import subprocess
import os
import sys
import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path


def check_ffmpeg():
    """检查 FFmpeg 是否安装"""
    try:
        result = subprocess.run(['ffmpeg', '-version'],
                              capture_output=True,
                              text=True,
                              timeout=5)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def trim_video_ffmpeg(input_path, output_path, start_time, duration):
    """
    使用 FFmpeg 剪辑视频

    Args:
        input_path: 输入视频路径
        output_path: 输出视频路径
        start_time: 开始时间（秒）
        duration: 持续时间（秒）

    Returns:
        bool: 是否成功
    """
    if not os.path.exists(input_path):
        print(f"错误：找不到输入文件 '{input_path}'")
        return False

    # GPS 数据将单独提取为 JSON 文件，视频可使用 MP4 格式输出
    if output_path.lower().endswith('.mp4'):
        print(f"[INFO] 检测到 MP4 输出格式")
        print(f"[INFO] GPS 数据将分离提取为独立的 JSON 文件")
        print(f"[INFO] 视频输出: {os.path.basename(output_path)}")

    # 检查输出目录是否存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print("步骤 1/2: 剪辑视频")
    print(f"{'='*70}\n")

    print(f"输入文件: {input_path}")
    print(f"输出文件: {output_path}")
    print(f"开始时间: {start_time} 秒")
    print(f"持续时间: {duration} 秒")

    # 构建 FFmpeg 命令
    # 只映射视频和音频流（GPS数据单独提取为JSON，不需要嵌入视频）
    # DJI视频流：0:0=视频, 0:1=音频, 0:2=DJI meta, 0:3=GPS, 0:4=Timecode
    command = [
        'ffmpeg',
        '-y',  # 覆盖输出文件
        '-ss', str(start_time),  # 开始时间（输入侧剪辑）
        '-i', input_path,  # 输入文件
        '-t', str(duration),  # 持续时间
        '-map', '0:0',  # 视频流
        '-map', '0:1',  # 音频流
        '-c', 'copy',  # 复制所有流，不重新编码
        output_path
    ]

    print(f"\n执行命令:")
    print(f"{' '.join(command)}\n")

    try:
        # 执行 FFmpeg 命令
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='ignore'
        )

        if result.returncode == 0:
            # 检查输出文件是否生成
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
                print(f"[SUCCESS] 视频剪辑成功!")
                print(f"  文件大小: {file_size:.2f} MB")
                return True, output_path  # 返回成功状态和实际文件路径
            else:
                print(f"[ERROR] 错误：输出文件未生成")
                return False, None
        else:
            print(f"[ERROR] FFmpeg 错误 (返回码: {result.returncode})")
            if result.stderr:
                print(f"错误信息:\n{result.stderr}")
            return False, None

    except Exception as e:
        print(f"[ERROR] 执行失败: {e}")
        return False, None


def extract_gps_to_json(input_video, start_time, duration, output_json):
    """
    使用 pyosmogps 提取 GPS 数据并保存为 JSON

    Args:
        input_video: 原始视频路径（用于提取完整GPS）
        start_time: 剪辑开始时间（秒）
        duration: 剪辑持续时间（秒）
        output_json: 输出 JSON 文件路径

    Returns:
        bool: 是否成功
    """
    print(f"\n{'='*70}")
    print("步骤 2/2: 提取GPS数据")
    print(f"{'='*70}\n")

    try:
        from pyosmogps import OsmoGps
        import warnings
        import io

        print(f"从原始视频提取 GPS: {input_video}")

        warnings.filterwarnings('ignore')
        old_stderr = sys.stderr
        sys.stderr = io.StringIO()

        try:
            # 从原始视频提取 GPS
            gps = OsmoGps([input_video], timezone_offset=8)
            gps.extract()

            all_lats = gps.get_latitude()
            all_lons = gps.get_longitude()
            all_alts = gps.get_altitude()

            if len(all_lats) == 0:
                print("[ERROR] 未找到 GPS 数据")
                return False

            print(f"[SUCCESS] 提取到 {len(all_lats)} 个 GPS 数据点")

            # 计算剪辑片段对应的 GPS 索引范围
            # DJI 视频的 GPS 采样率约为 50Hz
            gps_fps = 50.0
            start_index = int(start_time * gps_fps)
            end_index = int((start_time + duration) * gps_fps)

            # 确保索引在有效范围内
            start_index = max(0, min(start_index, len(all_lats) - 1))
            end_index = max(0, min(end_index, len(all_lats)))

            # 提取剪辑片段的 GPS 数据
            clip_lats = all_lats[start_index:end_index]
            clip_lons = all_lons[start_index:end_index]
            clip_alts = all_alts[start_index:end_index] if len(all_alts) > 0 else [0] * len(clip_lats)

            print(f"[SUCCESS] 提取剪辑片段 GPS: {len(clip_lats)} 个数据点")
            print(f"  起始索引: {start_index}")
            print(f"  结束索引: {end_index}")

            # 构建 GPS 数据结构
            gps_data = {
                'video_file': os.path.basename(input_video),
                'clip_start_time': start_time,
                'clip_duration': duration,
                'clip_end_time': start_time + duration,
                'gps_fps': gps_fps,
                'total_points': len(clip_lats),
                'data': []
            }

            # 为每个 GPS 点创建记录
            for i in range(len(clip_lats)):
                time_offset = i / gps_fps  # 相对于剪辑开始的时间
                absolute_time = start_time + time_offset  # 相对于原始视频开始的时间

                point = {
                    'index': start_index + i,
                    'time_offset': round(time_offset, 3),  # 剪辑内的相对时间（秒）
                    'absolute_time': round(absolute_time, 3),  # 原始视频中的绝对时间（秒）
                    'latitude': round(clip_lats[i], 8),
                    'longitude': round(clip_lons[i], 8),
                    'altitude': round(clip_alts[i], 2)
                }
                gps_data['data'].append(point)

            # 保存到 JSON 文件
            output_dir = os.path.dirname(output_json)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            with open(output_json, 'w', encoding='utf-8') as f:
                json.dump(gps_data, f, indent=2, ensure_ascii=False)

            print(f"[SUCCESS] GPS 数据已保存到: {output_json}")

            # 显示统计信息
            if clip_lats:
                print(f"\nGPS 统计:")
                print(f"  纬度范围: {min(clip_lats):.6f} ~ {max(clip_lats):.6f}")
                print(f"  经度范围: {min(clip_lons):.6f} ~ {max(clip_lons):.6f}")
                if clip_alts and any(alt > 0 for alt in clip_alts):
                    valid_alts = [alt for alt in clip_alts if alt > 0]
                    print(f"  海拔范围: {min(valid_alts):.1f}m ~ {max(valid_alts):.1f}m")

            return True

        finally:
            sys.stderr = old_stderr

    except ImportError:
        print("[ERROR] pyosmogps 未安装，无法提取 GPS")
        print("  安装方法: pip install pyosmogps")
        return False
    except Exception as e:
        print(f"[ERROR] GPS 提取失败: {e}")
        return False


def trim_video_with_gps(input_path, output_video, start_time, duration):
    """
    剪辑视频并提取GPS信息

    Args:
        input_path: 输入视频路径
        output_video: 输出视频路径
        start_time: 开始时间（秒）
        duration: 持续时间（秒）

    Returns:
        bool: 是否成功
    """
    print(f"\n{'='*70}")
    print("视频剪辑工具 - GPS分离提取模式")
    print(f"{'='*70}\n")

    # 步骤1: 剪辑视频
    success, actual_video_path = trim_video_ffmpeg(input_path, output_video, start_time, duration)
    if not success:
        return False

    # 使用实际的视频路径（与传入路径一致）
    if actual_video_path:
        output_video = actual_video_path

    # 步骤2: 提取GPS到JSON
    # 生成 JSON 文件路径（与视频同名）
    video_path = Path(output_video)
    json_path = str(video_path.with_suffix('.gps.json'))

    if not extract_gps_to_json(input_path, start_time, duration, json_path):
        print(f"\n[WARNING]  警告: GPS 提取失败，但视频已成功剪辑")

    print(f"\n{'='*70}")
    print("处理完成!")
    print(f"{'='*70}")
    print(f"输出视频: {output_video}")
    if os.path.exists(json_path):
        print(f"GPS数据: {json_path}")
    print()

    return True


def batch_trim_videos(input_path, output_dir, segments):
    """
    批量剪辑视频片段

    Args:
        input_path: 输入视频路径
        output_dir: 输出目录
        segments: 片段列表 [(start_time, duration, output_name), ...]

    Returns:
        int: 成功剪辑的数量
    """
    if not os.path.exists(input_path):
        print(f"错误：找不到输入文件 '{input_path}'")
        return 0

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    success_count = 0
    total_segments = len(segments)

    print(f"\n{'='*70}")
    print(f"批量剪辑模式 - 共 {total_segments} 个片段")
    print(f"{'='*70}\n")

    for i, (start_time, duration, output_name) in enumerate(segments, 1):
        output_path = os.path.join(output_dir, output_name)

        print(f"\n[{i}/{total_segments}] 正在处理...")

        if trim_video_with_gps(input_path, output_path, start_time, duration):
            success_count += 1
        else:
            print(f"片段 {i} 处理失败")

    print(f"\n{'='*70}")
    print(f"批量剪辑完成: {success_count}/{total_segments} 成功")
    print(f"{'='*70}\n")

    return success_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='视频剪辑工具 - GPS分离提取模式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:

  # 基本用法 - 从第10秒开始，剪辑30秒的视频
  python trim_video.py --input test.mp4 --output clip.mp4 --start 10 --duration 30

  # 从视频开头剪辑1分钟
  python trim_video.py --input test.mp4 --output clip.mp4 --start 0 --duration 60

  # 从第2分钟开始，剪辑45秒
  python trim_video.py --input test.mp4 --output clip.mp4 --start 120 --duration 45

  # 使用小数秒数
  python trim_video.py --input test.mp4 --output clip.mp4 --start 10.5 --duration 5.8

  # 批量剪辑（使用配置文件）
  python trim_video.py --input test.mp4 --output-dir clips/ --batch-config segments.json

输出文件:
  - 视频文件: output.mp4
  - GPS数据: output.gps.json (自动生成)

注意事项:
  - GPS 数据从原始视频提取，保存为独立的 JSON 文件
  - JSON 文件与视频文件同名，扩展名为 .gps.json
  - GPS 数据包含时间偏移、经纬度、海拔等信息
        """
    )

    parser.add_argument('--input', '-i', type=str, required=True,
                        help='输入视频路径')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='输出视频路径')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='输出目录（批量剪辑）')
    parser.add_argument('--start', '-s', type=float, default=None,
                        help='开始时间（秒）')
    parser.add_argument('--duration', '-d', type=float, default=None,
                        help='持续时间（秒）')
    parser.add_argument('--batch-config', type=str, default=None,
                        help='批量剪辑配置文件（JSON格式）')
    parser.add_argument('--info', action='store_true',
                        help='显示输入视频信息')

    args = parser.parse_args()

    # 检查 FFmpeg
    if not check_ffmpeg():
        print("错误：未找到 FFmpeg!")
        print("\n请先安装 FFmpeg:")
        print("  Windows: 下载并添加到 PATH")
        print("  Linux:   sudo apt install ffmpeg")
        print("  macOS:   brew install ffmpeg")
        sys.exit(1)

    # 显示视频信息
    if args.info:
        from pyosmogps import OsmoGps
        import warnings
        import io

        print(f"\n{'='*70}")
        print(f"视频信息: {args.input}")
        print(f"{'='*70}\n")

        # 基本文件信息
        if os.path.exists(args.input):
            file_size = os.path.getsize(args.input) / (1024*1024)
            print(f"文件大小: {file_size:.2f} MB")

        # GPS 信息
        try:
            warnings.filterwarnings('ignore')
            old_stderr = sys.stderr
            sys.stderr = io.StringIO()

            try:
                gps = OsmoGps([args.input], timezone_offset=8)
                gps.extract()
                lats = gps.get_latitude()

                if len(lats) > 0:
                    print(f"GPS数据点: {len(lats)}")
                    print(f"纬度范围: {min(lats):.6f} ~ {max(lats):.6f}")
                else:
                    print("GPS数据: 未找到")
            finally:
                sys.stderr = old_stderr
        except Exception as e:
            print(f"GPS检测: 失败 ({e})")

        sys.exit(0)

    # 批量剪辑模式
    if args.batch_config:
        if not args.output_dir:
            print("错误：批量剪辑需要指定 --output-dir")
            sys.exit(1)

        if not os.path.exists(args.batch_config):
            print(f"错误：找不到配置文件 '{args.batch_config}'")
            sys.exit(1)

        try:
            with open(args.batch_config, 'r', encoding='utf-8') as f:
                config = json.load(f)

            segments = []
            for item in config.get('segments', []):
                segments.append((
                    item['start'],
                    item['duration'],
                    item['output']
                ))

            batch_trim_videos(
                args.input,
                args.output_dir,
                segments
            )

        except json.JSONDecodeError as e:
            print(f"错误：配置文件格式错误 - {e}")
            sys.exit(1)
        except Exception as e:
            print(f"错误：{e}")
            sys.exit(1)

    # 单个剪辑模式
    else:
        if args.start is None or args.duration is None:
            print("错误：需要指定 --start 和 --duration 参数")
            print("\n使用 --info 查看视频信息")
            print("使用 --help 查看帮助")
            sys.exit(1)

        if args.output is None:
            # 自动生成输出文件名
            input_path = Path(args.input)
            output_name = f"{input_path.stem}_clip_{args.start}s_{args.duration}s{input_path.suffix}"
            args.output = str(input_path.parent / output_name)

        # 执行剪辑
        success = trim_video_with_gps(
            args.input,
            args.output,
            args.start,
            args.duration
        )

        sys.exit(0 if success else 1)
