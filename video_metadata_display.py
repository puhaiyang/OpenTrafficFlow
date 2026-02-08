import cv2
import numpy as np
import os
from datetime import datetime, timedelta
from moviepy import VideoFileClip
from PIL import Image, ImageDraw, ImageFont
import json
import subprocess
import struct
import re


def cv2ImgAddText(img, text, pos, textColor=(255, 255, 255), textSize=30):
    """在图像上添加中文文本"""
    if isinstance(img, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)

    font_paths = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/System/Library/Fonts/PingFang.ttc",
    ]

    font = None
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                font = ImageFont.truetype(font_path, textSize, encoding="utf-8")
                break
            except:
                continue

    if font is None:
        font = ImageFont.load_default()

    draw.text(pos, text, textColor, font=font)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def extract_dji_gps(video_path):
    """
    尝试从DJI视频中提取GPS信息
    DJI将GPS信息存储在protobuf格式的元数据流中
    """
    try:
        # 使用ffmpeg提取DJI元数据流
        cmd = [
            'ffmpeg', '-v', 'error', '-y',
            '-i', video_path,
            '-map', '0:2',  # DJI meta stream
            '-c', 'copy',
            '-f', 'data',
            '-'
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=10)

        if result.returncode == 0 and result.stdout:
            # 尝试在二进制数据中查找GPS模式
            data = result.stdout

            # DJI protobuf中可能包含GPS信息
            # 搜索可能的GPS坐标模式（浮点数）
            # 这是一个简化的方法，实际应该用protobuf解析

            # 尝试查找连续的浮点数模式
            import struct
            floats = []
            for i in range(0, len(data) - 8, 4):
                try:
                    val = struct.unpack('>f', data[i:i+4])[0]  # big-endian float
                    # GPS坐标范围：纬度 -90 到 90，经度 -180 到 180
                    if -90 <= val <= 90 or -180 <= val <= 180:
                        floats.append(val)
                except:
                    pass

            # 如果找到可能的坐标，返回第一对合理的值
            if len(floats) >= 2:
                # 简单启发式：找到第一个看起来像GPS坐标的对
                for i in range(len(floats) - 1):
                    lat, lon = floats[i], floats[i + 1]
                    if -90 <= lat <= 90 and -180 <= lon <= 180:
                        # 避免返回0,0（无效坐标）
                        if abs(lat) > 0.1 and abs(lon) > 0.1:
                            return lat, lon

    except Exception as e:
        pass

    return None, None


def extract_gps_from_video(video_path):
    """
    从视频中提取GPS信息
    支持多种方式：PyOsmoGPS（DJI Osmo Action）、ffprobe、exiftool、DJI元数据
    """
    gps_lat = None
    gps_lon = None

    # 方法1: 尝试使用PyOsmoGPS库（专门针对DJI Osmo Action 4/5）
    if gps_lat is None:
        try:
            import warnings
            warnings.filterwarnings('ignore')  # 抑制警告

            from pyosmogps import OsmoGps
            import io
            import sys

            # 捕获 stderr 以抑制 PyOsmoGPS 的解析错误输出
            old_stderr = sys.stderr
            sys.stderr = io.StringIO()

            try:
                # 创建OsmoGps实例（时区偏移8，中国时区UTC+8）
                gps = OsmoGps([video_path], timezone_offset=8)
                # 提取GPS数据
                gps.extract()

                # 获取第一个GPS坐标点（视频开始位置）
                if len(gps.get_latitude()) > 0 and len(gps.get_longitude()) > 0:
                    gps_lat = gps.get_latitude()[0]
                    gps_lon = gps.get_longitude()[0]
                    print(f"使用PyOsmoGPS成功提取GPS坐标")
            finally:
                # 恢复 stderr
                sys.stderr = old_stderr

        except ImportError:
            pass  # 静默失败，尝试其他方法
        except Exception:
            pass  # 静默失败，尝试其他方法

    # 方法2: 尝试使用ffprobe提取GPS（如果有安装ffmpeg）
    if gps_lat is None:
        try:
            cmd = [
                'ffprobe', '-v', 'error', '-show_entries',
                'format_tags=location:format_tags=com.apple.quicktime.location.ISO6709',
                '-of', 'json', video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and result.stdout:
                data = json.loads(result.stdout)
                # 尝试解析GPS信息
                format_tags = data.get('format', {}).get('tags', {})
                location = format_tags.get('location', '')

                if location:
                    # 解析GPS坐标（格式可能为：+27.9915+086.9323/）
                    if '+' in location and '/' in location:
                        parts = location.replace('/', '').split('+')
                        if len(parts) >= 3:
                            try:
                                lat = float(parts[1])
                                lon = float(parts[2])
                                gps_lat = lat
                                gps_lon = lon
                            except ValueError:
                                pass

        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, Exception):
            pass

    # 方法3: 尝试使用exiftool（如果安装了）
    if gps_lat is None:
        try:
            cmd = ['exiftool', '-json', '-GPSLatitude', '-GPSLongitude', video_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and result.stdout:
                data = json.loads(result.stdout)
                if len(data) > 0:
                    gps_data = data[0]
                    lat = gps_data.get('GPSLatitude')
                    lon = gps_data.get('GPSLongitude')
                    if lat and lon:
                        gps_lat = float(lat)
                        gps_lon = float(lon)
        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, Exception):
            pass

    # 方法4: 尝试从DJI元数据中提取GPS（二进制解析）
    if gps_lat is None:
        try:
            lat, lon = extract_dji_gps(video_path)
            if lat is not None and lon is not None:
                gps_lat = lat
                gps_lon = lon
        except Exception:
            pass

    return gps_lat, gps_lon


def extract_video_metadata(video_path):
    """
    从视频中提取元数据

    Args:
        video_path: 视频文件路径

    Returns:
        dict: 包含创建时间、GPS等信息的字典
    """
    metadata = {
        'creation_time': None,
        'gps_lat': None,
        'gps_lon': None,
        'duration': None,
        'has_gps': False
    }

    try:
        # 使用moviepy获取视频信息
        clip = VideoFileClip(video_path)
        metadata['duration'] = clip.duration

        # 尝试从文件名或系统获取创建时间
        if os.path.exists(video_path):
            creation_time = os.path.getmtime(video_path)
            metadata['creation_time'] = datetime.fromtimestamp(creation_time).strftime('%Y-%m-%d %H:%M:%S')

        clip.close()

        # 自动提取GPS信息
        gps_lat, gps_lon = extract_gps_from_video(video_path)
        if gps_lat is not None and gps_lon is not None:
            metadata['gps_lat'] = gps_lat
            metadata['gps_lon'] = gps_lon
            metadata['has_gps'] = True

    except Exception as e:
        print(f"提取视频元数据时出错: {e}")

    return metadata


def calculate_audio_volume(video_path, start_time, duration=0.1):
    """
    计算指定时间段的音频音量

    Args:
        video_path: 视频文件路径
        start_time: 开始时间（秒）
        duration: 采样时长（秒）

    Returns:
        float: 音量值（0-1范围）

    注意：此函数每次都会重新打开视频文件，效率较低。
      建议使用 AudioVolumeMonitor 类进行连续音量监控。
    """
    try:
        clip = VideoFileClip(video_path)

        # 检查是否有音频轨道
        if clip.audio is None:
            clip.close()
            return 0.0

        # 计算结束时间
        end_time = min(start_time + duration, clip.duration)
        if start_time >= clip.duration:
            clip.close()
            return 0.0

        # 方法1: 尝试使用subclip方法（moviepy 1.x）
        try:
            audio_segment = clip.audio.subclip(start_time, end_time)
        except AttributeError:
            # 方法2: moviepy 2.x 使用 with_start 和 with_end
            try:
                audio_segment = clip.audio.with_start(start_time).with_end(end_time)
            except AttributeError:
                # 方法3: 直接截取音频数组的对应部分
                # 获取整个音频并提取指定时间段
                full_audio = clip.audio.to_soundarray(fps=44100)
                total_samples = len(full_audio)
                total_duration = clip.audio.duration

                # 计算采样范围
                start_sample = int((start_time / total_duration) * total_samples)
                end_sample = int((end_time / total_duration) * total_samples)

                # 提取片段
                audio_array = full_audio[start_sample:end_sample]

                if len(audio_array) == 0:
                    clip.close()
                    return 0.0

                # 计算RMS
                if len(audio_array.shape) > 1:
                    rms = np.sqrt(np.mean(audio_array ** 2))
                else:
                    rms = np.sqrt(np.mean(audio_array ** 2))

                volume = min(rms * 2, 1.0)
                clip.close()
                return volume

        # 计算音量（RMS）
        audio_array = audio_segment.to_soundarray(fps=44100)

        if len(audio_array.shape) > 1:
            # 立体声，计算所有通道的平均值
            rms = np.sqrt(np.mean(audio_array ** 2))
        else:
            # 单声道
            rms = np.sqrt(np.mean(audio_array ** 2))

        # 归一化到0-1范围（假设最大RMS约为0.5）
        volume = min(rms * 2, 1.0)

        clip.close()
        return volume

    except Exception as e:
        # 静默失败，避免打印过多错误信息
        return 0.0


class AudioVolumeMonitor:
    """
    音频音量监控器 - 高效的连续音量采样

    在初始化时一次性加载整个音频数据到内存，
    然后可以快速计算任意时间段的音量。
    """

    def __init__(self, video_path):
        """
        初始化音量监控器

        Args:
            video_path: 视频文件路径
        """
        self.video_path = video_path
        self.audio_array = None
        self.audio_fps = 44100
        self.audio_duration = 0
        self.total_samples = 0
        self.has_audio = False

        self._load_audio()

    def _load_audio(self):
        """加载音频数据到内存"""
        try:
            clip = VideoFileClip(self.video_path)

            # 检查是否有音频轨道
            if clip.audio is None:
                clip.close()
                return

            # 将整个音频转换为numpy数组
            self.audio_array = clip.audio.to_soundarray(fps=self.audio_fps)
            self.audio_duration = clip.audio.duration
            self.total_samples = len(self.audio_array)
            self.has_audio = True

            clip.close()

        except Exception as e:
            print(f"加载音频数据时出错: {e}")
            self.has_audio = False

    def get_volume(self, start_time, duration=0.1):
        """
        获取指定时间段的音量

        Args:
            start_time: 开始时间（秒）
            duration: 采样时长（秒）

        Returns:
            float: 音量值（0-1范围）
        """
        if not self.has_audio or self.audio_array is None:
            return 0.0

        try:
            # 计算结束时间
            end_time = min(start_time + duration, self.audio_duration)
            if start_time >= self.audio_duration:
                return 0.0

            # 计算采样范围
            start_sample = int((start_time / self.audio_duration) * self.total_samples)
            end_sample = int((end_time / self.audio_duration) * self.total_samples)

            # 确保索引在有效范围内
            start_sample = max(0, min(start_sample, self.total_samples - 1))
            end_sample = max(0, min(end_sample, self.total_samples))

            # 提取片段
            audio_segment = self.audio_array[start_sample:end_sample]

            if len(audio_segment) == 0:
                return 0.0

            # 计算RMS（均方根）
            if len(audio_segment.shape) > 1:
                # 立体声或多声道 - 计算所有通道的平均值
                rms = np.sqrt(np.mean(audio_segment ** 2))
            else:
                # 单声道
                rms = np.sqrt(np.mean(audio_segment ** 2))

            # 归一化到0-1范围
            volume = min(rms * 2, 1.0)

            return volume

        except Exception as e:
            return 0.0


def format_timestamp(seconds):
    """将秒数格式化为时间戳"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 100)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:02d}"


def format_datetime(dt):
    """将datetime对象格式化为字符串"""
    return dt.strftime('%Y-%m-%d %H:%M:%S')


def extract_gps_trajectory(video_path):
    """
    提取视频的完整GPS轨迹（包括时间戳）

    Returns:
        dict: 包含时间戳列表和对应GPS坐标的字典
        {
            'timestamps': [datetime, ...],  # GPS时间戳
            'latitudes': [float, ...],      # 纬度
            'longitudes': [float, ...],     # 经度
            'altitudes': [float, ...],      # 海拔
            'has_gps': bool
        }
    """
    import warnings
    import io
    import sys

    trajectory = {
        'timestamps': [],
        'latitudes': [],
        'longitudes': [],
        'altitudes': [],
        'has_gps': False
    }

    try:
        warnings.filterwarnings('ignore')
        from pyosmogps import OsmoGps

        # 捕获 stderr 以抑制警告
        old_stderr = sys.stderr
        sys.stderr = io.StringIO()

        try:
            # 创建OsmoGps实例（时区偏移8，中国时区UTC+8）
            gps = OsmoGps([video_path], timezone_offset=8)
            # 提取GPS数据
            gps.extract()

            # 获取GPS轨迹数据
            lats = gps.get_latitude()
            lons = gps.get_longitude()
            alts = gps.get_altitude()

            # PyOsmoGPS的时间戳处理
            # GPS数据按时间顺序排列，采样频率约为50Hz（DJI Osmo Action 4）
            if len(lats) > 0:
                trajectory['latitudes'] = lats
                trajectory['longitudes'] = lons
                trajectory['altitudes'] = alts if len(alts) > 0 else [0] * len(lats)
                trajectory['has_gps'] = True

                # 生成时间戳列表（基于GPS采样频率）
                # DJI Osmo Action 4 GPS采样频率约为50fps
                gps_fps = 50.0  # GPS采样频率
                start_time = None  # 从视频元数据获取

                for i in range(len(lats)):
                    # 计算每个GPS点的时间偏移
                    time_offset = i / gps_fps
                    trajectory['timestamps'].append(time_offset)

        finally:
            sys.stderr = old_stderr

    except ImportError:
        pass
    except Exception:
        pass

    return trajectory


def get_gps_at_time(trajectory, video_start_time, current_video_time):
    """
    根据视频当前时间获取对应的GPS坐标

    Args:
        trajectory: GPS轨迹数据
        video_start_time: 视频开始录制的时间（datetime对象）
        current_video_time: 当前视频播放时间（秒）

    Returns:
        tuple: (latitude, longitude, altitude, real_datetime)
    """
    if not trajectory['has_gps'] or len(trajectory['latitudes']) == 0:
        return None, None, None, None

    # GPS采样频率（DJI Osmo Action 4约为50fps）
    gps_fps = 50.0

    # 计算对应的GPS索引
    gps_index = int(current_video_time * gps_fps)

    # 确保索引在有效范围内
    if gps_index >= len(trajectory['latitudes']):
        gps_index = len(trajectory['latitudes']) - 1

    lat = trajectory['latitudes'][gps_index]
    lon = trajectory['longitudes'][gps_index]
    alt = trajectory['altitudes'][gps_index] if gps_index < len(trajectory['altitudes']) else 0

    # 计算真实时间
    if video_start_time:
        from datetime import timedelta
        real_datetime = video_start_time + timedelta(seconds=current_video_time)
    else:
        real_datetime = None

    return lat, lon, alt, real_datetime


def process_video(video_path, output_path=None, display=True, save_output=True, gps_lat=None, gps_lon=None,
                  skip_frames=1, use_fast_codec=True):
    """
    处理视频文件，显示时间和元数据信息

    Args:
        video_path: 输入视频路径
        output_path: 输出视频路径（可选）
        display: 是否显示实时画面
        save_output: 是否保存输出视频
        gps_lat: 纬度（手动指定，优先级高于自动提取）
        gps_lon: 经度（手动指定，优先级高于自动提取）
        skip_frames: 跳帧处理（每N帧处理一次，默认1=每帧都处理）
        use_fast_codec: 使用快速编码器（H.264，默认True）
    """
    import warnings
    import io
    import sys

    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误：无法打开视频 {video_path}")
        return

    # 获取视频信息
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"视频信息: {width}x{height} @ {fps}fps, 总帧数: {total_frames}")

    # 性能优化信息
    if skip_frames > 1:
        print(f"性能优化: 跳帧处理（每{skip_frames}帧处理1帧），处理速度提升约{skip_frames}x")
    if use_fast_codec and save_output:
        print(f"性能优化: 使用H.264编码器（比mp4v快2-3倍）")

    # 提取视频元数据
    metadata = extract_video_metadata(video_path)

    # 提取完整GPS轨迹
    print("正在提取GPS轨迹...")
    gps_trajectory = extract_gps_trajectory(video_path)

    if gps_trajectory['has_gps']:
        print(f"成功提取 {len(gps_trajectory['latitudes'])} 个GPS数据点")
        metadata['has_gps'] = True
    elif gps_lat is not None and gps_lon is not None:
        metadata['gps_lat'] = gps_lat
        metadata['gps_lon'] = gps_lon
        metadata['has_gps'] = True
        print(f"使用手动指定的GPS坐标: {gps_lat}, {gps_lon}")
    else:
        print("注意：未检测到GPS信息")

    # 获取视频开始时间（用于显示真实时间）
    video_start_time = None
    try:
        if metadata['creation_time']:
            video_start_time = datetime.strptime(metadata['creation_time'], '%Y-%m-%d %H:%M:%S')
            print(f"视频开始时间: {metadata['creation_time']}")
    except:
        pass

    print(f"视频时长: {metadata['duration']:.2f}秒" if metadata['duration'] else "无法获取视频时长")

    # 初始化音频音量监控器（一次性加载音频数据到内存）
    print("正在加载音频数据...")
    audio_monitor = AudioVolumeMonitor(video_path)
    if audio_monitor.has_audio:
        print(f"音频数据加载成功，时长: {audio_monitor.audio_duration:.2f}秒")
    else:
        print("无音频轨道或音频加载失败")

    # 创建视频写入器
    video_writer = None
    if save_output and output_path:
        # 选择编码器
        if use_fast_codec:
            # 使用H.264编码器（需要安装）
            # Windows通常支持H.264
            fourcc = cv2.VideoWriter_fourcc(*'H264')  # 或使用 'avc1', 'X264'
            # 如果H264不可用，尝试其他编码器
        else:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if video_writer is None:
            print(f"警告：无法创建视频写入器，尝试使用备用编码器...")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if video_writer is not None:
            print(f"保存输出到: {output_path}")
        else:
            print(f"错误：无法创建视频写入器")
            save_output = False

    frame_count = 0
    processed_frame_count = 0
    audio_sampling_interval = 0.5 if skip_frames > 1 else 0.1  # 跳帧时降低采样频率
    last_audio_sample_time = -1
    current_volume = 0.0

    print("\n处理中... 按 'q' 提前退出")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # 跳帧处理：只处理指定的帧
        if frame_count % skip_frames != 0:
            continue

        processed_frame_count += 1

        # 计算当前时间戳
        current_time = frame_count / fps

        # 获取当前帧对应的GPS坐标
        current_lat, current_lon, current_alt, current_datetime = get_gps_at_time(
            gps_trajectory, video_start_time, current_time
        )

        # 定期采样音频（使用高效的音频监控器）
        if current_time - last_audio_sample_time >= audio_sampling_interval:
            current_volume = audio_monitor.get_volume(current_time, duration=audio_sampling_interval)
            last_audio_sample_time = current_time

        # 创建信息面板背景
        panel_height = 220  # 增加面板高度以容纳更多信息
        panel = np.zeros((panel_height, width, 3), dtype=np.uint8)
        panel[:] = (0, 0, 0)  # 黑色背景

        # 绘制信息文本
        y_offset = 35
        line_height = 35

        # 真实时间（GPS时间）
        if current_datetime:
            time_str = format_datetime(current_datetime)
            cv2.putText(panel, f"Time: {time_str}", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        else:
            # 如果没有真实时间，显示播放时长
            timestamp = format_timestamp(current_time)
            cv2.putText(panel, f"Time: {timestamp}", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        y_offset += line_height

        # GPS信息（实时更新）
        if current_lat is not None and current_lon is not None:
            gps_text = f"GPS: {current_lat:.6f}, {current_lon:.6f}"
            cv2.putText(panel, gps_text, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            y_offset += line_height

            # 显示海拔
            alt_text = f"Alt: {current_alt:.1f}m"
            cv2.putText(panel, alt_text, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(panel, "GPS: No Signal", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)
        y_offset += line_height

        # 音量信息（带可视化条）
        volume_percent = current_volume * 100
        volume_text = f"Vol: {volume_percent:.1f}%"
        cv2.putText(panel, volume_text, (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

        # 绘制音量条
        bar_width = 300
        bar_height = 20
        bar_x = 250
        bar_y = y_offset - 15

        # 音量条背景
        cv2.rectangle(panel, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                     (64, 64, 64), -1)

        # 音量条前景（根据音量大小变色）
        volume_bar_width = int(bar_width * current_volume)
        if current_volume < 0.3:
            bar_color = (0, 255, 0)  # 绿色
        elif current_volume < 0.7:
            bar_color = (0, 165, 255)  # 橙色
        else:
            bar_color = (0, 0, 255)  # 红色

        cv2.rectangle(panel, (bar_x, bar_y), (bar_x + volume_bar_width, bar_y + bar_height),
                     bar_color, -1)

        # 将信息面板叠加到帧上
        frame[:panel_height, :] = panel

        # 写入输出视频
        if video_writer:
            video_writer.write(frame)

        # 显示进度
        if frame_count % 30 == 0:
            progress = frame_count / total_frames * 100
            speedup = skip_frames if skip_frames > 1 else 1
            print(f"已处理 {frame_count}/{total_frames} 帧 ({progress:.1f}%) - 速度提升: {speedup}x")

        # 显示画面
        if display:
            # 添加帧数信息
            frame_info = f"Frame: {frame_count}/{total_frames}"
            cv2.putText(frame, frame_info, (width - 300, panel_height + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Video Metadata Display", frame)

            # 按 'q' 退出，按 ' ' 暂停
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n用户停止处理")
                break
            elif key == ord(' '):
                print("暂停 - 按任意键继续...")
                cv2.waitKey(0)

    # 释放资源
    cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()

    actual_speedup = frame_count / processed_frame_count if processed_frame_count > 0 else 1
    print(f"\n处理完成！")
    print(f"  总帧数: {frame_count}")
    print(f"  处理帧数: {processed_frame_count}")
    if skip_frames > 1:
        print(f"  实际加速: {actual_speedup:.1f}x")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='视频元数据显示与输出')
    parser.add_argument('--video', type=str, required=True,
                        help='输入视频路径')
    parser.add_argument('--output', type=str, default='metadata_output.mp4',
                        help='输出视频路径（默认: metadata_output.mp4）')
    parser.add_argument('--no-display', action='store_true',
                        help='不显示实时视频')
    parser.add_argument('--no-save', action='store_true',
                        help='不保存输出视频')
    parser.add_argument('--gps-lat', type=float, default=None,
                        help='GPS纬度（例如: 39.9042）')
    parser.add_argument('--gps-lon', type=float, default=None,
                        help='GPS经度（例如: 116.4074）')
    parser.add_argument('--skip-frames', type=int, default=1,
                        help='跳帧处理（每N帧处理1帧，默认1=每帧都处理。设置为5可提升5倍速度）')
    parser.add_argument('--slow-codec', action='store_true',
                        help='使用较慢的mp4v编码器（默认使用快速H.264编码器）')

    args = parser.parse_args()

    # 检查输入视频是否存在
    if not os.path.exists(args.video):
        print(f"错误：找不到视频文件 '{args.video}'！")
        print("\n使用示例:")
        print("  python video_metadata_display.py --video test.mp4")
        print("  python video_metadata_display.py --video test.mp4 --output result.mp4")
        print("  python video_metadata_display.py --video test.mp4 --gps-lat 39.9042 --gps-lon 116.4074")
        print("  python video_metadata_display.py --video test.mp4 --no-display --no-save")
        print("\n性能优化:")
        print("  python video_metadata_display.py --video test.mp4 --skip-frames 5  # 5倍速")
        print("  python video_metadata_display.py --video test.mp4 --skip-frames 10 --no-display  # 10倍速")
    else:
        process_video(
            video_path=args.video,
            output_path=args.output if not args.no_save else None,
            display=not args.no_display,
            save_output=not args.no_save,
            gps_lat=args.gps_lat,
            gps_lon=args.gps_lon,
            skip_frames=args.skip_frames,
            use_fast_codec=not args.slow_codec
        )
