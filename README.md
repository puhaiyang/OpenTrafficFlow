# OpenTrafficFlow
Make traffic faster and safer.

文章地址：
- [《小白都能看懂的——车牌检测与识别(最新版YOLO26快速入门)》](https://blog.csdn.net/puhaiyang/article/details/157846503)



预览效果：
![traffic.gif](preview/traffic.gif)



## 数据集合格式
### CCPD-2019数据集

它的图片内容在ccpd_开头的目录下，详细文件格式如下：

> ll CCPD_Datasets/CCPD/puhaiyang___CCPD2019/CCPD2019/ccpd_base/
> -rw-rw-r-- 1 1000 1000  80339  2月  6  2019 '0212703544062-86_91-244&503_460&597-465&574_259&593_258&514_464&495-0_0_16_25_25_29_13-140-72.jpg'

(ultralytics-env) [root@xg-ragflow-node1 OpenTrafficFlow]# ll CCPD_Datasets/CCPD/puhaiyang___CCPD2019/CCPD2019/
总用量 50748
drwxrwxr-x 2 1000 1000 30953472  2月  4 16:36 ccpd_base
drwxrwxr-x 2 1000 1000  2822144  2月  4 16:16 ccpd_blur
drwxrwxr-x 2 1000 1000  6844416  2月  4 16:14 ccpd_challenge
drwxrwxr-x 2 1000 1000  1388544  2月  4 16:17 ccpd_db
drwxrwxr-x 2 1000 1000  2920448  2月  4 16:10 ccpd_fn
drwxrwxr-x 2 1000 1000    69632  2月  4 16:17 ccpd_np
drwxrwxr-x 2 1000 1000  1376256  2月  4 16:18 ccpd_rotate
drwxrwxr-x 2 1000 1000  4194304  2月  4 16:39 ccpd_tilt
drwxrwxr-x 2 1000 1000  1359872  2月  4 16:15 ccpd_weather
-rw-rw-r-- 1 1000 1000     1061  8月 25  2018 LICENSE
-rw-rw-r-- 1 1000 1000     4022  8月 25  2018 README.md
drwxrwxr-x 2 1000 1000     4096  2月  4 16:17 splits


### CCPD-2020数据集
(ultralytics-env) [root@xg-ragflow-node1 OpenTrafficFlow]# ll CCPD_Datasets/CCPD2020/puhaiyang___CCPD2020/CCPD2020/ccpd_green/
总用量 1864
drwxr-xr-x 2 root root 811008  2月  4 15:55 test
drwxr-xr-x 2 root root 929792  2月  4 15:55 train
drwxr-xr-x 2 root root 167936  2月  4 15:55 val

格式为：
(ultralytics-env) [root@xg-ragflow-node1 OpenTrafficFlow]# head -n 10| ll CCPD_Datasets/CCPD2020/puhaiyang___CCPD2020/CCPD2020/ccpd_green/train/ 
总用量 450032
-rw-r--r-- 1 root root  92934  2月  4 15:55 '00360785590278-91_265-311&485_406&524-406&524_313&520_311&485_402&489-0_0_3_24_28_24_31_33-117-16.jpg'
-rw-r--r-- 1 root root 133527  2月  4 15:55 '00373372395833-90_96-276&514_387&548-387&548_276&547_276&516_384&514-0_0_3_26_25_31_33_32-157-19.jpg'


# 安装
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install opencv-python
pip install pillow

# 转化
> python convert_all_ccpd_to_yolo.py --all --source ./CCPD_Datasets --target ./YOLO_Data --copy


# 车牌提取
https://github.com/sirius-ai/LPRNet_Pytorch

---

## Android 部署 - 模型转换为 NCNN 格式

本项目支持将训练好的模型转换为 NCNN 格式，用于在 Android 设备上部署。

### 模型文件说明

- **YOLO 检测模型**: `weights/best.pt` - 用于车牌检测
- **LPRNet 识别模型**: `weights/Final_LPRNet_model.pth` - 用于车牌字符识别

---

### 一、YOLO 模型转换 (best.pt → NCNN)

使用 Ultralytics 内置的导出功能，一键转换 YOLO 模型到 NCNN 格式。

#### 转换步骤

**1. 安装 Ultralytics**

```bash
pip install ultralytics
```

**2. 运行转换代码**

```python
from ultralytics import YOLO

# 加载 YOLO 模型
model = YOLO('weights/best.pt')

# 导出为 NCNN 格式
model.export(format='ncnn')
```

**3. 获取转换结果**

转换完成后，会生成以下文件：
- `best.pt.ncnn.param` - NCNN 参数文件（约 26 KB）
- `best.pt.ncnn.bin` - NCNN 权重文件（约 9.2 MB）

#### 常用导出参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `format` | 导出格式，设置为 `'ncnn'` | - |
| `imgsz` | 输入图像尺寸 | 原始训练尺寸 |
| `half` | 是否使用 FP16 精度 | `False` |
| `simplify` | 是否简化 ONNX 模型 | `True` |

#### 注意事项

- YOLO 模型会先转换为 ONNX 格式，然后再转换为 NCNN 格式
- 确保已安装 `onnx` 和 `onnx-simplifier` 依赖
- 转换后的 NCNN 模型可以在 Android 上使用 NCNN 框架加载

---

### 二、LPRNet 模型转换 (Final_LPRNet_model.pth → NCNN)

LPRNet 模型需要先转换为 ONNX 格式，然后使用在线工具转换为 NCNN。

#### 步骤 1：PyTorch → ONNX

使用项目提供的转换脚本：

```bash
python convert_simple.py
```

该脚本会：
- 加载 `weights/Final_LPRNet_model.pth`
- 设置正确的类别数 (class_num=68)
- 导出为 `weights_ncnn/lprnet.onnx`

**输入维度**: `[1, 3, 24, 94]` - (batch, channels, height, width)

#### 步骤 2：ONNX → NCNN (在线转换)

使用 **ConvertModel** 在线工具进行转换：

**1. 访问转换网站**

```
https://convertmodel-1256200149.cos-website.ap-nanjing.myqcloud.com/
```

**2. 选择转换格式**
- 输入格式: **ONNX**
- 输出格式: **NCNN**

**3. 上传并转换**
- 上传: `lprnet.onnx` (约 1.71 MB)
- 点击转换按钮

**4. 下载结果**
- 下载生成的 `.param` 和 `.bin` 文件
- 重命名为 `lprnet.param` 和 `lprnet.bin`

#### 其他在线转换工具

| 工具名称 | 地址 | 特点 |
|---------|------|------|
| **PNNX 在线工具** | https://pnnx.pchar.cn/ | 本地运行，不上传模型，数据隐私安全 |
| **ToolForge** | https://toolforge.homes/tools/svc-ncnn-v1 | ONNX → NCNN 专用转换器 |

---

### 三、Android 集成

转换完成后，将生成的 NCNN 文件复制到 Android 项目的 `assets` 目录：

```
Android项目/app/src/main/assets/
├── yolo_plate.param      # YOLO 检测模型
├── yolo_plate.bin        # YOLO 权重
├── lprnet.param          # LPRNet 识别模型
└── lprnet.bin            # LPRNet 权重
```

#### NCNN Android 库

- **下载地址**: https://github.com/Tencent/ncnn/releases
- **选择版本**: `ncnn-android-vulkan.zip` (支持 GPU 加速)
- **集成方式**: 将 `.so` 文件放入 `app/src/main/jniLibs/` 对应架构目录

#### 支持的 CPU 架构

```
jniLibs/
├── arm64-v8a/
│   └── libncnn.so
└── armeabi-v7a/
    └── libncnn.so
```

---

### 四、参考资源

- [Ultralytics 文档 - 导出模型](https://docs.ultralytics.com/modes/export/)
- [NCNN GitHub](https://github.com/Tencent/ncnn)
- [NCNN 使用指南](https://github.com/Tencent/ncnn/wiki/use-ncnn-with-pytorch-or-onnx)


# 上传违章举报
https://github.com/zai-org/Open-AutoGLM

> ./run.bat --base-url https://open.bigmodel.cn/api/paas/v4 --model "autoglm-phone" --apikey "YOUR_KEY" --device-id "ip:port" "打开微信，找到成都交警公众号，进去之后点蓉e行，进入蓉e行中，点击交通违法举报，点击侵走非机动车道，选择视频导入模式"

---

# 了解详细

获取自动违章举报软件的最新信息，欢迎扫码入群：

![wx_group.png](preview/wx_group.png)


---

## 各地交通违法举报平台汇总

以下是全国各地交通违法举报平台的信息汇总：


| 地区 | 平台类型 | 访问方式/搜索关键词 |
|------|----------|---------------------|
| 北京 | 微信公众号 | 关注"北京交警"微信公众号，点击底部菜单"随手拍"按钮 |
| 天津 | 小程序 | 搜索"天津交通违法举报"小程序 |
| 上海 | APP | 使用"上海交警"APP进行举报 |
| 广东 | 微信公众号 | 关注"广东交警"微信公众号，在公众号内搜索"举报"，参考第一篇文章查看各城市举报方式 |
| 广西 | 微信公众号 | 关注"广西警微发布"微信公众号，点击底部菜单"违法举报"入口 |
| 浙江 | 微信公众号 | 关注"浙江高速交警"微信公众号，点击"高速出行"菜单，选择"随手拍"子菜单 |
| 云南 | 微信公众号 | 关注当地交警微信公众号（如"昆明交警"），搜索"举报"，查看弹出的举报方法或在公众号内搜索"举报"找到介绍举报方法的文章 |
| 贵州 | APP | 通过"贵州交警"APP进行举报 |
| 湖南 | 微信公众号 | 关注"湖南高速警察"微信公众号，点击底部菜单"违法举报"入口 |
| 江西 | 微信公众号 | 关注当地交警微信公众号（如"南昌交警"），搜索"举报"，参考介绍举报方法的文章 |
| 福建 | 微信公众号 | 关注当地交警微信公众号（如"厦门交警"），搜索"举报"，参考介绍举报方法的文章 |
| 江苏 | 微信公众号 | 关注当地交警微信公众号（如"南京交警"），搜索"举报"，底部菜单有"交通违法"入口，进入后选择"交通违法举报" |
| 安徽 | 微信公众号 | 关注"安徽公安交警在线"微信公众号，在第二篇文章中搜索"举报"查看省市级举报渠道 |
| 湖北 | 微信公众号 | 关注"湖北交警"微信公众号，从底部菜单选择"违法举报" |
| 河南 | 微信公众号 | 关注市级交警微信公众号（如"郑州交警"），搜索"举报"查看举报方法（如"西安交警"的"随手拍"） |
| 四川 | 微信公众号+小程序 | 关注市级交警微信公众号，搜索"举报"查看方法（如通过菜单入口进入"蓉e行"小程序） |
| 重庆 | 微信公众号 | 关注"重庆交巡警"微信公众号，点击"随警快办" → "用户中心" → "交巡警专区" → "违法举报" |
| 河北 | 微信公众号 | 关注市级高速交警微信公众号（如"石家庄高速交警微发布"），从底部菜单选择"违法举报" |
| 陕西 | 微信公众号 | 关注市级交警微信公众号，搜索"举报"查看方法（如"西安交警"的"随手拍"） |
| 宁夏 | 微信公众号 | 关注市级交警微信公众号，搜索"举报"查看方法（如"银川交警"的"随手拍"） |
| 海南 | 邮箱+微信公众号 | 发送邮件至 hnjj12123@126.com；关注"海南交警"微信公众号，搜索"举报"查看详细要求 |
| 辽宁 | 微信公众号 | 关注市级交警微信公众号，搜索"举报"查看举报方法；例如关注"大连公安交警"微信公众号，点击菜单栏"警民互动"，然后选择"违法举报" |
| 吉林 | 微信公众号 | 关注市级交警微信公众号，搜索"举报"查看举报方法；例如关注"通化公安交警支队"微信公众号，点击菜单栏"警民互动"，然后从子菜单选择"随手拍" |
| 内蒙古 | 微信公众号 | 关注"内蒙古交警"微信公众号，点击底部菜单栏"随手拍" |
| 甘肃 | 微信公众号 | 关注市级交警微信公众号，搜索"举报"查看举报方法；例如关注"兰州公安交警"微信公众号，点击菜单栏"个人中心"，然后从子菜单选择"随手拍小程序" |
| 新疆 | 微信公众号 | 关注"新疆交警"微信公众号，点击底部菜单栏"随手拍" |
| 西藏 | — | 暂未找到相关线上举报渠道 |
| 黑龙江 | — | 暂未找到相关线上举报渠道 |
| 青海 | — | 暂未找到相关线上举报渠道 |
| 山西 | — | 暂未找到相关线上举报渠道 |
