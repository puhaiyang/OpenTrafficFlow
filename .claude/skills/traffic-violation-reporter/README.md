# Traffic Violation Reporter Skill

Automate traffic violation reporting through Chengdu Police official WeChat account using Open-AutoGLM and ADB.

## Overview

This skill automates the complete workflow of reporting traffic violations:
1. Starts mitmproxy with GPS coordinate mocking
2. Uses Open-AutoGLM to control phone via ADB
3. Navigates WeChat to Chengdu Police official account
4. Reports traffic violation with video evidence
5. Submits the report

## Quick Start

### Prerequisites

1. **Android device** with USB debugging enabled
2. **ADB installed** and device connected
3. **Open-AutoGLM** submodule (already added to project)
4. **mitmproxy installed**: `pip install mitmproxy`
5. **AutoGLM model service** running (local or remote)
6. **WeChat installed** on the device
7. **ADB Keyboard** installed and enabled on device

### Basic Usage

The simplest way to use this skill is through the main integration script:

```bash
cd .claude/skills/traffic-violation-reporter/scripts
python report_traffic_violation.py
```

This will:
- Check prerequisites (ADB, mitmproxy, Open-AutoGLM)
- Load violation data from `examples/video1.json`
- Start mitmproxy with GPS mocking
- Execute AutoGLM task
- Submit the report

### Custom Parameters

```bash
python report_traffic_violation.py \
  --json-path examples/video1.json \
  --location "人民南路一段和天府广场" \
  --base-url http://localhost:8000/v1 \
  --model autoglm-phone-9b
```

## Individual Components

### 1. Start Mitmproxy Only

```bash
python start_mitmproxy.py --json-path examples/video1.json
```

### 2. Execute AutoGLM Task Only

```bash
python execute_autoglm_task.py \
  --json-path examples/video1.json \
  --location "沈阳路西段和润郎路" \
  --base-url http://localhost:8000/v1
```

## JSON File Format

The video metadata JSON file should contain:

```json
{
  "time": "2026/3/25 8:46:31",
  "position": "104.065551,30.471042",
  "license_plate_no": "川AA91637",
  "against_type": "侵走非机动车道",
  "car_type": "绿牌汽车（小型）"
}
```

## Phone Setup

### Enable USB Debugging

1. Go to `Settings → About Phone`
2. Tap `Version Number` 7 times to enable Developer Mode
3. Go to `Settings → Developer Options`
4. Enable `USB Debugging`
5. On some phones, also enable `USB Debugging (Security Settings)`

### Install ADB Keyboard

1. Download ADB Keyboard: https://github.com/senzhk/ADBKeyBoard/blob/master/ADBKeyboard.apk
2. Install on your Android device
3. Go to `Settings → Language & Input → Virtual Keyboard`
4. Enable `ADB Keyboard`

### Configure Phone Proxy

For mitmproxy to intercept location requests:

1. Connect phone to same WiFi as computer
2. Go to `Settings → Wi-Fi → Long press on network → Modify network`
3. Set proxy to `localhost:8080` (or your computer's IP)
4. Install mitmproxy CA certificate (displayed on first request)

## Model Service

### Local Deployment

If you have a GPU with 24GB+ VRAM:

```bash
pip install vllm
python3 -m vllm.entrypoints.openai.api_server \
  --served-model-name autoglm-phone-9b \
  --allowed-local-media-path / \
  --mm-encoder-tp-mode data \
  --mm-processor-cache-type shm \
  --mm-processor-kwargs "{\"max_pixels\":5000000}" \
  --max-model-len 25480 \
  --chat-template-content-format string \
  --limit-mm-per-prompt "{\"image\":10}" \
  --model zai-org/AutoGLM-Phone-9B \
  --port 8000
```

### Remote Services

Use these third-party services (no local GPU needed):

**BigModel:**
```bash
--base-url https://open.bigmodel.cn/api/paas/v4 \
--model autoglm-phone \
--apikey your-api-key
```

**ModelScope:**
```bash
--base-url https://api-inference.modelscope.cn/v1 \
--model ZhipuAI/AutoGLM-Phone-9B \
--apikey your-api-key
```

## Workflow Details

The skill performs these steps:

1. **Load JSON Data**: Read violation information from JSON file
2. **Generate Mock Script**: Create mitmproxy script with GPS coordinates
3. **Start Mitmproxy**: Launch mitmproxy to intercept location requests
4. **Generate AutoGLM Task**: Create detailed task description
5. **Execute Task**: Run AutoGLM to automate phone operations
6. **Cleanup**: Stop mitmproxy after completion

## AutoGLM Task Description

The generated AutoGLM task includes:

```
打开微信，找到成都交警公众号，进去之后点蓉e行，
进入蓉e行中，点击交通违法举报，点击【侵走非机动车道】，
选择【视频导入模式】。在弹出的地图界面中，随便点一个地点
并点击【确定】按钮，再点击界面下方的【确定举报地点】按钮。
之后在新界面中，违法时间进行点击，时间选择时间为【2026年3月25日】，
之后点击确定；在弹出的"请选择违法时分"界面中，将时间中的
小时滚动到【08时】，分钟滚动到【36分】，再点击【确认】按钮
进行确认。之后，在行为描述界面中，输入："2026年3月25日，
8点36分，在'沈阳路西段和润郎路'路口，绿牌汽车（小型），
车牌号为：【川AA91637】侵走非机动车道，此行为给行人和非机动
车带来了严重的安全隐患，望审核后进行相应的处罚，谢谢。"之后，
点击界面中的上传证据，在弹框中选择【从相册选择】，再选择相册中
的顶部最左侧的第一个视频，之后点击【完成】。返回到交通违法举报
界面后，点击【提交举报信息】。
```

## Troubleshooting

### ADB Device Not Found

```bash
adb kill-server
adb start-server
adb devices
```

Ensure USB debugging is enabled and you clicked "Allow USB debugging" on the phone.

### Mitmproxy Not Intercepting

- Check phone proxy settings
- Verify mitmproxy is running: `ps aux | grep mitmproxy`
- Check if CA certificate is installed on phone

### AutoGLM Task Fails

- Verify model service is running: `curl http://localhost:8000/v1/models`
- Check task description format
- Ensure phone screen is unlocked and bright
- Verify WeChat is logged in

### Video Upload Fails

- Check if video exists in phone gallery
- Ensure video is in supported format (MP4)
- Verify gallery permissions

## Safety Notes

1. **Verify Accuracy**: Double-check violation details before submission
2. **Legal Compliance**: Ensure reports follow local traffic laws
3. **Privacy**: Protect personal information in reports
4. **Test First**: Test in non-production environment first

## License

This skill is part of the OpenTrafficFlow project. Use responsibly and in compliance with local laws and regulations.

## Support

For issues or questions:
1. Check Open-AutoGLM documentation: `Open-AutoGLM/README.md`
2. Verify mitmproxy installation: `mitmproxy --version`
3. Test ADB connection: `adb devices`
4. Check model service status: Model service logs