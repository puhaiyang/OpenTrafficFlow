#!/usr/bin/env python3
"""
AutoGLM task executor for traffic violation reporting.
Generates and executes AutoGLM tasks using Open-AutoGLM framework.
"""

import subprocess
import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime


def parse_violation_time(time_str):
    """
    Parse violation time from JSON format.

    Args:
        time_str: Time string like "2026/3/25 8:46:31"

    Returns:
        dict: Parsed date, hour, minute, and time string
    """
    try:
        dt = datetime.strptime(time_str, "%Y/%m/%d %H:%M:%S")

        return {
            'date_str': f"{dt.year}年{dt.month}月{dt.day}日",
            'hour': f"{dt.hour:02d}",
            'minute': f"{dt.minute:02d}",
            'time_str': f"{dt.year}年{dt.month}月{dt.day}日，{dt.hour}点{dt.minute}分"
        }
    except Exception as e:
        print(f"Error parsing time: {e}")
        return None


def load_violation_data(json_path):
    """
    Load violation data from JSON file.

    Args:
        json_path: Path to JSON file containing video metadata

    Returns:
        dict: Violation data or None if error
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Parse time
        time_data = parse_violation_time(data.get('time', ''))

        return {
            'time': data.get('time', ''),
            'position': data.get('position', ''),
            'license_plate_no': data.get('license_plate_no', ''),
            'against_type': data.get('against_type', ''),
            'car_type': data.get('car_type', ''),
            'parsed_time': time_data
        }
    except Exception as e:
        print(f"Error loading violation data: {e}")
        return None


def generate_violation_description(violation_data, location="沈阳路西段和润郎路"):
    """
    Generate formatted violation description for the report.

    Args:
        violation_data: Dict with violation details
        location: Location description (default: 沈阳路西段和润郎路)

    Returns:
        str: Formatted violation description
    """
    if violation_data['parsed_time'] is None:
        return None

    time_str = violation_data['parsed_time']['time_str']
    license_plate = violation_data['license_plate_no']
    car_type = violation_data['car_type']
    against_type = violation_data['against_type']

    description = (
        f"{time_str}，在'{location}'路口，{car_type}，车牌号为："
        f"【{license_plate}】{against_type}，此行为给行人和非机动车"
        f"带来了严重的安全隐患，望审核后进行相应的处罚，谢谢。"
    )

    return description


def generate_autoglm_task(violation_data, location="沈阳路西段和润郎路"):
    """
    Generate AutoGLM task description for traffic violation reporting.

    Args:
        violation_data: Dict with violation details
        location: Location description (default: 沈阳路西段和润郎路)

    Returns:
        str: Complete AutoGLM task description or None if error
    """
    if violation_data['parsed_time'] is None:
        print("❌ Error: Could not parse violation time")
        return None

    violation_type = violation_data['against_type']
    date_str = violation_data['parsed_time']['date_str']
    hour_str = violation_data['parsed_time']['hour']
    minute_str = violation_data['parsed_time']['minute']
    description = generate_violation_description(violation_data, location)

    if description is None:
        return None

    task = (
        f"打开微信，找到成都交警公众号，进去之后点蓉e行，进入蓉e行中，"
        f"点击交通违法举报，点击【{violation_type}】，选择【视频导入模式】。"
        f"在弹出的地图界面中，随便点一个地点并点击【确定】按钮，"
        f"再点击界面下方的【确定举报地点】按钮。之后在新界面中，"
        f"违法时间进行点击，时间选择时间为【{date_str}】，之后点击确定；"
        f"在弹出的\"请选择违法时分\"界面中，将时间中的小时滚动到【{hour_str}时】，"
        f"分钟滚动到【{minute_str}分】，再点击【确认】按钮进行确认。"
        f"之后，在行为描述界面中，输入：\"{description}\"。之后，"
        f"点击界面中的上传证据，在弹框中选择【从相册选择】，"
        f"再选择相册中的顶部最左侧的第一个视频，之后点击【完成】。"
        f"返回到交通违法举报界面后，点击【提交举报信息】。"
    )

    return task


def execute_autoglm_task(task_description, base_url, model_name,
                       opengl_path=None, device_id=None, verbose=True):
    """
    Execute AutoGLM task with given parameters.

    Args:
        task_description: AutoGLM task description
        base_url: Model API URL
        model_name: Model name
        opengl_path: Path to Open-AutoGLM directory
        device_id: ADB device ID (optional)
        verbose: Print detailed output

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Determine Open-AutoGLM path
        if opengl_path is None:
            script_dir = Path(__file__).resolve().parent
            project_root = script_dir.parent.parent.parent
            opengl_path = project_root / "Open-AutoGLM"

        if not opengl_path.exists():
            print(f"❌ Error: Open-AutoGLM not found at {opengl_path}")
            return False

        print(f"📱 Using Open-AutoGLM at: {opengl_path}")

        # Build command
        cmd = [
            sys.executable,
            str(opengl_path / "main.py"),
            "--base-url", base_url,
            "--model", model_name,
            task_description
        ]

        if device_id:
            cmd.extend(["--device-id", device_id])

        if verbose:
            cmd.append("--verbose")

        print(f"🚀 Executing AutoGLM task...")
        print(f"🔧 Base URL: {base_url}")
        print(f"🤖 Model: {model_name}")
        if device_id:
            print(f"📱 Device ID: {device_id}")
        print()

        # Execute command
        result = subprocess.run(
            cmd,
            cwd=str(opengl_path),
            capture_output=False,
            text=True
        )

        if result.returncode == 0:
            print("✓ AutoGLM task completed successfully")
            return True
        else:
            print(f"❌ AutoGLM task failed with return code: {result.returncode}")
            return False

    except Exception as e:
        print(f"❌ Error executing AutoGLM task: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Execute AutoGLM task for traffic violation reporting'
    )
    parser.add_argument(
        '--json-path',
        default='examples/video1.json',
        help='Path to video metadata JSON file (default: examples/video1.json)'
    )
    parser.add_argument(
        '--location',
        default='沈阳路西段和润郎路',
        help='Location description for report (default: 沈阳路西段和润郎路)'
    )
    parser.add_argument(
        '--base-url',
        default='http://localhost:8000/v1',
        help='Model API URL (default: http://localhost:8000/v1)'
    )
    parser.add_argument(
        '--model',
        default='autoglm-phone-9b',
        help='Model name (default: autoglm-phone-9b)'
    )
    parser.add_argument(
        '--opengl-path',
        help='Path to Open-AutoGLM directory (default: project root/Open-AutoGLM)'
    )
    parser.add_argument(
        '--device-id',
        help='ADB device ID (default: auto-detect)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help='Enable verbose output (default: True)'
    )

    args = parser.parse_args()

    print(f"📋 Traffic Violation Reporter - AutoGLM Task Executor")
    print()

    # Get project root (3 levels up from script location)
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent.parent

    # Convert relative path to absolute
    json_path = project_root / args.json_path

    print(f"📄 JSON file: {json_path}")
    print(f"📍 Location: {args.location}")
    print()

    # Load violation data
    violation_data = load_violation_data(json_path)
    if violation_data is None:
        print("❌ Error: Could not load violation data")
        return 1

    print("📊 Violation Data:")
    print(f"  Time: {violation_data['time']}")
    print(f"  Position: {violation_data['position']}")
    print(f"  License Plate: {violation_data['license_plate_no']}")
    print(f"  Violation Type: {violation_data['against_type']}")
    print(f"  Vehicle Type: {violation_data['car_type']}")
    print()

    # Generate AutoGLM task
    task = generate_autoglm_task(violation_data, args.location)

    if task is None:
        print("❌ Error: Could not generate AutoGLM task")
        return 1

    print("📝 Generated AutoGLM Task:")
    print(task)
    print()

    # Confirm before execution
    response = input("✅ Ready to execute? (y/n): ").strip().lower()
    if response != 'y':
        print("❌ Execution cancelled")
        return 0

    # Execute AutoGLM task
    success = execute_autoglm_task(
        task,
        args.base_url,
        args.model,
        args.opengl_path,
        args.device_id,
        args.verbose
    )

    if success:
        print("\n🎉 Traffic violation report submitted successfully!")
        print(f"📋 Report details:")
        print(f"  License: {violation_data['license_plate_no']}")
        print(f"  Type: {violation_data['against_type']}")
        print(f"  Location: {args.location}")
        print(f"  Time: {violation_data['time']}")
        return 0
    else:
        print("\n❌ Failed to submit traffic violation report")
        return 1


if __name__ == '__main__':
    sys.exit(main())