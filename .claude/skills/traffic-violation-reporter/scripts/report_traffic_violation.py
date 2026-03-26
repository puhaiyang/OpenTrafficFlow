#!/usr/bin/env python3
"""
Main integration script for traffic violation reporting.
Combines mitmproxy GPS mocking and AutoGLM automation.
"""

import subprocess
import sys
import os
import argparse
import json
import time
import signal
from pathlib import Path
from datetime import datetime


class TrafficViolationReporter:
    """Main class for traffic violation reporting automation."""

    def __init__(self, json_path, location="沈阳路西段和润郎路",
                 base_url="http://localhost:8000/v1",
                 model_name="autoglm-phone-9b"):
        self.json_path = Path(json_path)
        self.location = location
        self.base_url = base_url
        self.model_name = model_name
        self.mitmproxy_process = None
        self.script_dir = Path(__file__).resolve().parent

        # Get project root
        self.project_root = self.script_dir.parent.parent.parent

        # Paths to helper scripts
        self.start_mitmproxy_script = self.script_dir / "start_mitmproxy.py"
        self.execute_autoglm_script = self.script_dir / "execute_autoglm_task.py"

    def load_violation_data(self):
        """Load and display violation data from JSON file."""
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            print("📊 Violation Data:")
            print(f"  Time: {data.get('time', 'N/A')}")
            print(f"  Position: {data.get('position', 'N/A')}")
            print(f"  License Plate: {data.get('license_plate_no', 'N/A')}")
            print(f"  Violation Type: {data.get('against_type', 'N/A')}")
            print(f"  Vehicle Type: {data.get('car_type', 'N/A')}")
            print()

            return data
        except Exception as e:
            print(f"❌ Error loading violation data: {e}")
            return None

    def check_prerequisites(self):
        """Check if all prerequisites are met."""
        print("🔍 Checking prerequisites...")

        issues = []

        # Check JSON file
        if not self.json_path.exists():
            issues.append(f"JSON file not found: {self.json_path}")

        # Check helper scripts
        if not self.start_mitmproxy_script.exists():
            issues.append(f"Mitmproxy script not found: {self.start_mitmproxy_script}")

        if not self.execute_autoglm_script.exists():
            issues.append(f"AutoGLM script not found: {self.execute_autoglm_script}")

        # Check Open-AutoGLM
        opengl_path = self.project_root / "Open-AutoGLM"
        if not opengl_path.exists():
            issues.append(f"Open-AutoGLM not found at: {opengl_path}")

        # Check ADB connection
        try:
            result = subprocess.run(['adb', 'devices'],
                                  capture_output=True, text=True)
            if 'device' not in result.stdout:
                issues.append("No ADB device found. Check connection and USB debugging.")
        except FileNotFoundError:
            issues.append("ADB not found. Install Android platform-tools.")

        # Check mitmproxy
        try:
            result = subprocess.run(['mitmproxy', '--version'],
                                  capture_output=True, text=True)
            if result.returncode != 0:
                issues.append("Mitmproxy not installed. Install with: pip install mitmproxy")
        except FileNotFoundError:
            issues.append("Mitmproxy not found. Install with: pip install mitmproxy")

        if issues:
            print("❌ Prerequisites check failed:")
            for issue in issues:
                print(f"  - {issue}")
            return False

        print("✓ All prerequisites met")
        print()
        return True

    def start_mitmproxy(self):
        """Start mitmproxy with GPS mocking in background."""
        try:
            print(f"🚀 Starting mitmproxy with GPS mocking...")

            cmd = [
                sys.executable,
                str(self.start_mitmproxy_script),
                '--json-path', str(self.json_path.relative_to(self.project_root))
            ]

            self.mitmproxy_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Wait a bit for startup
            time.sleep(3)

            if self.mitmproxy_process.poll() is None:
                print(f"✓ Mitmproxy started (PID: {self.mitmproxy_process.pid})")
                return True
            else:
                print("❌ Mitmproxy failed to start")
                return False

        except Exception as e:
            print(f"❌ Error starting mitmproxy: {e}")
            return False

    def stop_mitmproxy(self):
        """Stop mitmproxy process."""
        if self.mitmproxy_process:
            print("🛑 Stopping mitmproxy...")
            self.mitmproxy_process.terminate()

            try:
                self.mitmproxy_process.wait(timeout=5)
                print("✓ Mitmproxy stopped")
            except subprocess.TimeoutExpired:
                print("⚠️  Mitmproxy did not stop gracefully, killing...")
                self.mitmproxy_process.kill()
                self.mitmproxy_process.wait()

    def execute_autoglm(self):
        """Execute AutoGLM task."""
        try:
            print(f"📱 Executing AutoGLM task...")

            cmd = [
                sys.executable,
                str(self.execute_autoglm_script),
                '--json-path', str(self.json_path.relative_to(self.project_root)),
                '--location', self.location,
                '--base-url', self.base_url,
                '--model', self.model_name
            ]

            result = subprocess.run(
                cmd,
                capture_output=False,
                text=True
            )

            return result.returncode == 0

        except Exception as e:
            print(f"❌ Error executing AutoGLM: {e}")
            return False

    def run(self):
        """Run the complete traffic violation reporting workflow."""
        print(f"📋 Traffic Violation Reporter - Starting Report")
        print(f"📁 Project: {self.project_root}")
        print(f"📄 JSON: {self.json_path}")
        print(f"📍 Location: {self.location}")
        print()

        # Check prerequisites
        if not self.check_prerequisites():
            return 1

        # Load and display violation data
        violation_data = self.load_violation_data()
        if violation_data is None:
            return 1

        # Confirm before starting
        response = input("✅ Ready to start reporting? (y/n): ").strip().lower()
        if response != 'y':
            print("❌ Operation cancelled")
            return 0

        try:
            # Start mitmproxy
            if not self.start_mitmproxy():
                return 1

            print()
            print("⏳ Mitmproxy is running with GPS mocking")
            print("   Ensure your phone's proxy is configured correctly")
            print("   Press Enter to continue with AutoGLM task...")
            input()

            print()
            print("🤖 Executing AutoGLM task...")
            print("   Monitor your phone for automated actions")
            print()

            # Execute AutoGLM task
            success = self.execute_autoglm()

            print()
            if success:
                print("🎉 Traffic violation report submitted successfully!")
                print(f"📋 Report Summary:")
                print(f"   License: {violation_data['license_plate_no']}")
                print(f"   Type: {violation_data['against_type']}")
                print(f"   Location: {self.location}")
                print(f"   Time: {violation_data['time']}")
                return 0
            else:
                print("❌ Failed to submit traffic violation report")
                return 1

        except KeyboardInterrupt:
            print("\n\n⚠️  Operation interrupted by user")
            return 1

        finally:
            # Always stop mitmproxy
            self.stop_mitmproxy()


def main():
    parser = argparse.ArgumentParser(
        description='Automate traffic violation reporting via Chengdu Police WeChat'
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

    args = parser.parse_args()

    # Get project root for path resolution
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent.parent
    json_path = project_root / args.json_path

    reporter = TrafficViolationReporter(
        json_path=json_path,
        location=args.location,
        base_url=args.base_url,
        model_name=args.model
    )

    sys.exit(reporter.run())


if __name__ == '__main__':
    sys.exit(main())