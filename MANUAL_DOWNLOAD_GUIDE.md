# CCPD 数据集手动下载指南（Windows 备选方案）

当自动下载脚本持续卡住时，可以使用本手动下载方案。

---

## 方案 1: 使用浏览器直接下载

### CCPD2020 数据集（推荐，865MB）

1. 访问 OpenXLab 网站: https://openxlab.org.cn/datasets/OpenDataLab/CCPD

2. 登录账号后，导航到文件列表

3. 下载 `CCPD2020.zip` 文件

4. 保存到: `F:\project\OpenTrafficFlow\CCPD_Datasets\CCPD2020\`

5. 使用以下命令解压：
```bash
cd F:\project\OpenTrafficFlow\CCPD_Datasets\CCPD2020
tar -xf CCPD2020.zip
```

### CCPD 数据集（12.6GB）

**警告**: Windows 系统通过浏览器下载此文件容易超时，推荐使用方案 2。

---

## 方案 2: 使用专业下载工具（推荐）

### 推荐工具

1. **Internet Download Manager (IDM)** - 付费，但下载大文件最稳定
2. **Free Download Manager (FDM)** - 免费，支持断点续传
3. **aria2** - 开源命令行工具

### 使用 IDM/Free Download Manager 下载

#### 步骤 1: 获取下载链接

1. 访问 https://openxlab.org.cn/datasets/OpenDataLab/CCPD
2. 登录并找到文件列表
3. 右键点击 `CCPD2019.tar.xz` → 复制下载链接

#### 步骤 2: 使用下载工具

**IDM**:
```
1. 打开 IDM
2. 点击 "添加 URL"
3. 粘贴下载链接
4. 设置保存路径: F:\project\OpenTrafficFlow\CCPD_Datasets\CCPD\
5. 开始下载
```

**Free Download Manager**:
```
1. 打开 FDM
2. 点击 "新建下载"
3. 粘贴下载链接
4. 设置保存路径
5. 开始下载
```

#### 步骤 3: 验证下载完整性

下载完成后，检查文件大小：
- CCPD2020.zip: 应约为 865 MB
- CCPD2019.tar.xz: 应约为 12.6 GB

#### 步骤 4: 解压文件

```bash
# CCPD2020
cd F:\project\OpenTrafficFlow\CCPD_Datasets\CCPD2020
python -m zipfile -e CCPD2020.zip .

# CCPD2019
cd F:\project\OpenTrafficFlow\CCPD_Datasets\CCPD
python -m tarfile -e CCPD2019.tar.xz .
```

或使用 7-Zip / WinRAR 右键解压。

---

## 方案 3: 使用 aria2 命令行工具

### 安装 aria2

```bash
# 使用 Scoop 安装
scoop install aria2

# 或使用 Chocolatey
choco install aria2
```

### 下载文件

```bash
# 创建下载链接文本文件 download.txt
echo "https://openxlab.org.cn/..." > download.txt

# 使用 aria2 下载（支持断点续传）
aria2c -c -x 16 -s 16 -d "F:\project\OpenTrafficFlow\CCPD_Datasets\CCPD" -i download.txt

# 参数说明：
# -c: 断点续传
# -x 16: 最大连接数 16
# -s 16: 单任务最大连接数 16
# -d: 保存目录
```

如果下载中断，重新运行相同命令即可继续下载。

---

## Windows Defender 配置（重要）

### 添加排除项

1. 打开 **Windows 安全中心**
2. 点击 **病毒和威胁防护**
3. 在"病毒和威胁防护设置"下，点击 **管理设置**
4. 滚动到"排除项"，点击 **添加或删除排除项**
5. 点击 **添加排除项** → **文件夹**
6. 添加: `F:\project\OpenTrafficFlow\CCPD_Datasets`

这样可以避免 Windows Defender 在下载/解压大文件时进行干扰。

---

## 解压后验证

### CCPD2020

解压后应该有：
```
CCPD2020/
├── test/
├── train/
└── README.md
```

### CCPD2019

解压后应该有：
```
CCPD2019/
├── test/
├── train/
└── ...
```

---

## 常见问题

### Q: 下载速度很慢怎么办？

A:
1. 尝试更换网络（有线 > WiFi）
2. 关闭 VPN 和代理
3. 使用专业下载工具（IDM/FDM）
4. 选择非高峰时段下载

### Q: 下载中断怎么办？

A:
1. 使用支持断点续传的工具（IDM/FDM/aria2）
2. 重新运行下载命令/任务
3. 避免中途关闭电脑或休眠

### Q: 解压失败怎么办？

A:
1. 检查文件大小是否完整
2. 使用 7-Zip 而非 Windows 自带解压
3. 重新下载不完整的文件

---

## 推荐流程

对于 Windows 用户，推荐按以下优先级尝试：

1. ✅ **首选**: 使用 `download_ccpd_windows.py` 下载 CCPD2020
2. ✅ **次选**: 使用 IDM/FDM 手动下载大文件
3. ✅ **备选**: 使用 aria2 命令行工具
4. ❌ **不推荐**: 直接使用浏览器下载 12GB+ 文件

---

如有问题，请检查：
- 网络连接是否稳定
- 硬盘空间是否充足（至少 30GB）
- Windows Defender 是否已添加排除项
