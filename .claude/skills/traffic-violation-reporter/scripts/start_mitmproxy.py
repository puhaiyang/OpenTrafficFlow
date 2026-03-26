#!/usr/bin/env python3
"""
Mitmproxy startup script for traffic violation reporting.
Starts mitmproxy with GPS coordinate mocking for location-based requests.
"""

import subprocess
import sys
import os
import argparse
import json
from pathlib import Path


def load_gps_coordinates(json_path):
    """
    Load GPS coordinates from video metadata JSON file.

    Args:
        json_path: Path to the JSON file containing video metadata

    Returns:
        tuple: (longitude, latitude) or (None, None) if not found
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        position = data.get('position', '')
        if ',' in position:
            jd, wd = position.split(',')
            return jd.strip(), wd.strip()
        return None, None
    except Exception as e:
        print(f"Error loading GPS coordinates: {e}")
        return None, None


def generate_mock_script(jd, wd, output_path):
    """
    Generate mitmproxy script with GPS coordinates.

    Args:
        jd: Longitude coordinate
        wd: Latitude coordinate
        output_path: Path to write the generated script
    """
    script_content = f'''"""Modify HTTP/HTTPS query parameters for traffic violation reporting."""

from mitmproxy import http


def request(flow: http.HTTPFlow) -> None:
    """Intercept and modify location requests."""
    if flow.request.url.startswith("https://rex.cdnet110.com/api-gateway-wfjb/rex-wfjb/userEeport/transform/address"):
        flow.request.query["jd"] = "{jd}"
        flow.request.query["wd"] = "{wd}"
    if flow.request.url.startswith("https://rex.cdnet110.com/api-gateway-wfjb/rex-wfjb/userEeport/transform"):
        flow.request.query["jd"] = "{jd}"
        flow.request.query["wd"] = "{wd}"
'''

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(script_content)

    print(f"✓ Generated mock script at: {output_path}")
    print(f"  GPS Coordinates: JD={jd}, WD={wd}")


def start_mitmproxy(script_path, port=8080, web_port=8081):
    """
    Start mitmproxy with the generated script.

    Args:
        script_path: Path to the mitmproxy script
        port: Proxy port (default: 8080)
        web_port: Web interface port (default: 8081)
    """
    try:
        # Check if mitmproxy is installed
        result = subprocess.run(['mitmproxy', '--version'],
                              capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ Error: mitmproxy is not installed")
            print("Install it with: pip install mitmproxy")
            return False

        print(f"🚀 Starting mitmproxy...")
        print(f"  Script: {script_path}")
        print(f"  Proxy port: {port}")
        print(f"  Web interface: http://localhost:{web_port}")

        # Start mitmproxy in the background
        cmd = [
            'mitmweb',
            '-s', str(script_path),
            '-p', str(port),
            '--web-port', str(web_port),
            '--set', 'ssl_insecure=true'
        ]

        # Start the process
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE)

        print(f"✓ Mitmproxy started with PID: {process.pid}")
        print(f"  Press Ctrl+C to stop")

        return process

    except FileNotFoundError:
        print("❌ Error: mitmproxy command not found")
        print("Install it with: pip install mitmproxy")
        return None
    except Exception as e:
        print(f"❌ Error starting mitmproxy: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Start mitmproxy with GPS mocking for traffic violation reporting'
    )
    parser.add_argument(
        '--json-path',
        default='examples/video1.json',
        help='Path to video metadata JSON file (default: examples/video1.json)'
    )
    parser.add_argument(
        '--mock-script',
        default='skill/mock_http.py',
        help='Path to output mock script (default: skill/mock_http.py)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8080,
        help='Proxy port (default: 8080)'
    )
    parser.add_argument(
        '--web-port',
        type=int,
        default=8081,
        help='Web interface port (default: 8081)'
    )

    args = parser.parse_args()

    # Get project root (3 levels up from script location)
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent.parent

    # Convert relative paths to absolute
    json_path = project_root / args.json_path
    mock_script_path = project_root / args.mock_script

    print(f"📋 Traffic Violation Reporter - Mitmproxy Startup")
    print(f"📁 Project root: {project_root}")
    print(f"📄 JSON file: {json_path}")
    print()

    # Load GPS coordinates
    jd, wd = load_gps_coordinates(json_path)
    if jd is None or wd is None:
        print("❌ Error: Could not load GPS coordinates from JSON file")
        return 1

    # Generate mock script
    generate_mock_script(jd, wd, mock_script_path)

    # Start mitmproxy
    process = start_mitmproxy(mock_script_path, args.port, args.web_port)

    if process is None:
        return 1

    try:
        # Keep the script running
        process.wait()
    except KeyboardInterrupt:
        print("\n🛑 Stopping mitmproxy...")
        process.terminate()
        process.wait()
        print("✓ Mitmproxy stopped")

    return 0


if __name__ == '__main__':
    sys.exit(main())