---
name: traffic-violation-reporter
description: Automate traffic violation reporting through Chengdu Police official WeChat account using Open-AutoGLM and ADB. This skill handles the complete workflow: starting mitmproxy for location mocking, launching WeChat, navigating to Chengdu Police official account, entering Rong-E-Xing service, selecting traffic violation reporting type, choosing video import mode, selecting violation location with mocked GPS coordinates, setting violation time, inputting violation description, uploading video evidence, and submitting the report. Use this skill whenever the user mentions traffic violation reporting, automated reporting, WeChat police account reporting, Rong-E-Xing service, or wants to automatically report traffic violations detected from dashcam or security camera footage.
compatibility: Requires Open-AutoGLM submodule, ADB connection to Android device, mitmproxy installed, and video metadata JSON file with GPS coordinates.
---

# Traffic Violation Reporter Skill

This skill automates the process of reporting traffic violations through the Chengdu Police official WeChat account (成都交警) using Open-AutoGLM to control the phone via ADB.

## Overview

The skill performs these steps in sequence:
1. **Start mitmproxy** with GPS coordinate mocking script
2. **Launch Open-AutoGLM** to control phone operations
3. **Navigate WeChat** to Chengdu Police official account
4. **Access Rong-E-Xing** (蓉e行) service
5. **Report traffic violation** with pre-filled information
6. **Upload video evidence** from gallery
7. **Submit the report**

## Prerequisites

Before using this skill, ensure you have:

1. **Android device** with USB debugging enabled and ADB connection
2. **Open-AutoGLM** submodule installed in the project
3. **mitmproxy** installed (`pip install mitmproxy`)
4. **Video metadata JSON** file with GPS coordinates (e.g., `examples/video1.json`)
5. **WeChat installed** on the Android device
6. **ADB Keyboard** installed and enabled on the device
7. **Video file** in device gallery for upload evidence

## Skill Workflow

### Step 1: Extract Report Data from JSON

Read the video metadata JSON file to extract:
- **GPS coordinates** (position field) for location mocking
- **License plate number** (license_plate_no)
- **Violation type** (against_type)
- **Vehicle type** (car_type)
- **Timestamp** (time field)

Example JSON structure:
```json
{
  "time": "2026/3/25 8:46:31",
  "position": "104.065551,30.471042",
  "license_plate_no": "川AA91637",
  "against_type": "侵走非机动车道",
  "car_type": "绿牌汽车（小型）"
}
```

### Step 2: Start Mitmproxy with Location Mocking

Start mitmproxy with the GPS mocking script:
```bash
cd project/root
mitmweb -s skill/mock_http.py
```

The `mock_http.py` script intercepts location requests and returns the GPS coordinates from the JSON file.

### Step 3: Generate AutoGLM Task Description

Create a detailed task description for Open-AutoGLM that includes:
- Exact navigation path through WeChat
- Specific UI elements to tap
- Text input for violation description
- Time selection steps
- Video upload steps
- Final submission

### Step 4: Execute AutoGLM Task

Run Open-AutoGLM with the generated task description:
```bash
cd Open-AutoGLM
python main.py --base-url {MODEL_URL} --model "autoglm-phone-9b" "{TASK_DESCRIPTION}"
```

### Step 5: Cleanup

After successful submission:
- Stop mitmproxy server
- Log the report completion with details

## Using the Skill

### Basic Usage

When the user asks to report a traffic violation:

1. **Verify JSON file path** (default: `examples/video1.json`)
2. **Extract violation data** from the JSON
3. **Update mitmproxy script** with GPS coordinates
4. **Start mitmproxy** in background
5. **Generate AutoGLM task** with violation details
6. **Execute the task** via Open-AutoGLM
7. **Monitor progress** and handle any errors
8. **Stop mitmproxy** after completion

### AutoGLM Task Template

The AutoGLM task should follow this structure:

```
打开微信，找到成都交警公众号，进去之后点蓉e行，进入蓉e行中，点击交通违法举报，点击【侵走非机动车道】，选择【视频导入模式】。在弹出的地图界面中，随便点一个地点并点击【确定】按钮，再点击界面下方的【确定举报地点】按钮。之后在新界面中，违法时间进行点击，时间选择时间为【{VIOLATION_DATE}】，之后点击确定；在弹出的"请选择违法时分"界面中，将时间中的小时滚动到【{VIOLATION_HOUR}时】，分钟滚动到【{VIOLATION_MINUTE}分】，再点击【确认】按钮进行确认。之后，在行为描述界面中，输入："{VIOLATION_DESCRIPTION}"。之后，点击界面中的上传证据，在弹框中选择【从相册选择】，再选择相册中的顶部最左侧的第一个视频，之后点击【完成】。返回到交通违法举报界面后，点击【提交举报信息】。
```

Replace the placeholders with actual data:
- `{VIOLATION_DATE}`: Date from JSON (e.g., "2026年3月25日")
- `{VIOLATION_HOUR}`: Hour from JSON (e.g., "08")
- `{VIOLATION_MINUTE}`: Minute from JSON (e.g., "36")
- `{VIOLATION_DESCRIPTION}`: Formatted description with license plate, location, vehicle type, and violation type

### Violation Description Format

Format the violation description as:
```
{YEAR}年{MONTH}月{DAY}日，{HOUR}点{MINUTE}分，在'{LOCATION}'路口，{CAR_TYPE}，车牌号为：【{LICENSE_PLATE}】{VIOLATION_TYPE}，此行为给行人和非机动车带来了严重的安全隐患，望审核后进行相应的处罚，谢谢。
```

Example:
```
2026年3月25日，8点36分，在'沈阳路西段和润郎路'路口，绿牌汽车（小型），车牌号为：【川AA91637】侵走非机动车道，此行为给行人和非机动车带来了严重的安全隐患，望审核后进行相应的处罚，谢谢。
```

## Error Handling

### Common Issues

1. **ADB connection fails**:
   - Check `adb devices` output
   - Ensure USB debugging is enabled
   - Verify data cable supports data transfer

2. **Mitmproxy not intercepting**:
   - Check phone proxy settings (should point to mitmproxy)
   - Verify CA certificate is installed on phone
   - Check if script is loaded correctly

3. **AutoGLM task fails**:
   - Verify model service is running
   - Check task description format
   - Ensure screen is bright and unlocked
   - Verify WeChat is logged in

4. **Video upload fails**:
   - Check if video exists in gallery
   - Ensure video is in supported format
   - Verify gallery permissions

### Recovery Steps

If the task fails mid-way:
1. Take screenshot of current screen state
2. Analyze where the task stopped
3. Generate continuation task from current state
4. Resume AutoGLM with new task description
5. Log the partial progress

## Advanced Configuration

### Custom JSON File Path

Users can specify a custom JSON file path:
```python
json_path = "path/to/custom/video_data.json"
```

### Model Service Configuration

AutoGLM can use different model services:
- **Local deployment**: `http://localhost:8000/v1`
- **BigModel**: `https://open.bigmodel.cn/api/paas/v4`
- **ModelScope**: `https://api-inference.modelscope.cn/v1`

### Batch Processing

For multiple violations, the skill can:
1. Read multiple JSON files from a directory
2. Process each file sequentially
3. Generate individual reports
4. Log completion status for each

## Monitoring and Logging

The skill provides:
- **Progress updates** at each step
- **Error messages** with recovery suggestions
- **Completion confirmation** with report details
- **Time tracking** for performance monitoring

## Safety Considerations

1. **Verify accuracy**: Always double-check violation details before submission
2. **Legal compliance**: Ensure all reports follow local traffic laws
3. **Privacy**: Protect personal information when generating reports
4. **Test environment**: Test in a non-production environment first

## Troubleshooting

Check the bundled scripts for:
- `scripts/start_mitmproxy.py`: Mitmproxy startup automation
- `scripts/update_mock_coords.py`: GPS coordinate updater
- `scripts/execute_autoglm_task.py`: AutoGLM task executor

These scripts handle common issues and provide detailed logging for debugging.