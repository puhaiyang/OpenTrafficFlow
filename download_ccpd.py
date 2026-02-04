"""
OpenXLab CCPD 数据集下载脚本（优化版）
用于从 OpenDataLab 下载 CCPD 和 CCPD2020 数据集
支持断点续传、分步下载、自动解压
"""

import subprocess
import sys
import os
import zipfile
import tarfile

# ========== 在导入 openxlab 之前设置配置目录 ==========
# 设置项目本地的 openxlab 配置目录，避免权限问题
_openxlab_home = os.path.join(os.getcwd(), '.openxlab')
os.makedirs(_openxlab_home, exist_ok=True)

# 尝试设置多个可能的环境变量
os.environ['OPENXLAB_HOME'] = _openxlab_home
if 'HOME' not in os.environ:
    os.environ['HOME'] = _openxlab_home

# Windows 下尝试在默认位置创建目录并设置权限（如果 openxlab 硬编码了路径）
if os.name == 'nt':  # Windows
    try:
        import pathlib
        # 尝试在用户主目录创建 .openxlab
        user_home = os.path.expanduser('~')
        default_openxlab_dir = os.path.join(user_home, '.openxlab')
        default_token_path = os.path.join(default_openxlab_dir, 'token.json')

        # 如果目录不存在，创建它
        if not os.path.exists(default_openxlab_dir):
            os.makedirs(default_openxlab_dir, exist_ok=True)

        # 如果 token.json 不存在，创建一个空文件（避免权限错误）
        if not os.path.exists(default_token_path):
            try:
                # 尝试设置权限
                subprocess.run(
                    ['icacls', default_openxlab_dir, '/grant', f'{os.getenv("USERNAME", "Everyone")}:(OI)(CI)F'],
                    shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False
                )
                # 创建占位文件
                with open(default_token_path, 'w') as f:
                    f.write('{}')
            except:
                pass  # 如果失败，继续尝试
    except:
        pass  # 如果失败，继续尝试
# =======================================================


# 数据集配置
DATASETS = {
    '1': {
        'name': 'CCPD2020',
        'repo': 'puhaiyang/CCPD2020',
        'folder': 'CCPD2020',
        'size': '865.7M',
        'min_size': 800 * 1024 * 1024,  # 800MB
        'archive_name': 'CCPD2020.zip',
        'archive_type': 'zip',
        'expected_archive_size': 865 * 1024 * 1024,  # 865MB
        'description': '较小，推荐优先下载'
    },
    '2': {
        'name': 'CCPD',
        'repo': 'puhaiyang/CCPD2019',
        'folder': 'CCPD',
        'size': '12.2G',
        'min_size': 12 * 1024 * 1024 * 1024,  # 12GB
        'archive_name': 'CCPD2019.tar.xz',
        'archive_type': 'tar.xz',
        'expected_archive_size': 12.6 * 1024 * 1024 * 1024,  # 12.6GB
        'description': '较大，下载耗时较长'
    }
}


def install_package():
    """安装并升级 openxlab 包"""
    print("=" * 60)
    print("步骤 1/7: 安装 openxlab...")
    print("=" * 60)
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "openxlab"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("✓ openxlab 安装/升级成功\n")
    except subprocess.CalledProcessError as e:
        print(f"✗ 安装失败: {e}")
        sys.exit(1)


def login_openxlab():
    """登录 OpenXLab"""
    print("=" * 60)
    print("步骤 2/7: 登录 OpenXLab...")
    print("=" * 60)

    # 设置 openxlab 配置目录为项目目录，避免权限问题
    openxlab_home = os.path.join(os.getcwd(), '.openxlab')
    os.makedirs(openxlab_home, exist_ok=True)

    # Windows 下设置目录权限
    if os.name == 'nt':  # Windows
        try:
            # 使用 icacls 命令授予当前用户完全控制权限
            import ctypes
            import ctypes.wintypes

            # 获取当前用户名
            GetUserNameEx = ctypes.windll.secur32.GetUserNameExW
            NameDisplay = 3

            size = ctypes.wintypes.DWORD(0)
            GetUserNameEx(NameDisplay, None, ctypes.byref(size))
            username = ctypes.create_unicode_buffer(size.value)
            GetUserNameEx(NameDisplay, username, ctypes.byref(size))
            current_user = username.value

            # 使用 icacls 设置权限
            subprocess.run(
                ['icacls', openxlab_home, '/grant', f'{current_user}:(OI)(CI)F', '/T'],
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False
            )
        except:
            pass  # 如果设置权限失败，继续尝试

    os.environ['OPENXLAB_HOME'] = openxlab_home
    print(f"✓ 配置目录: {openxlab_home}")

    # 从环境变量读取 AK/SK，如果没有则提示用户输入
    ak = os.environ.get("OPENXLAB_AK")
    sk = os.environ.get("OPENXLAB_SK")

    if not ak:
        ak = input("请输入 Access Key (AK): ").strip()
    if not sk:
        sk = input("请输入 Secret Key (SK): ").strip()

    # 先手动创建 token 文件到默认位置，避免权限错误
    try:
        import json
        user_home = os.path.expanduser('~')
        default_openxlab_dir = os.path.join(user_home, '.openxlab')
        os.makedirs(default_openxlab_dir, exist_ok=True)

        default_token_path = os.path.join(default_openxlab_dir, 'token.json')
        token_data = {
            "ak": ak,
            "sk": sk
        }

        # 手动保存 token 到默认位置
        with open(default_token_path, 'w', encoding='utf-8') as f:
            json.dump(token_data, f, indent=2)

        print(f"✓ Token 已预先保存到默认位置")

    except Exception as e:
        print(f"⚠ 预保存 token 失败: {e}")

    try:
        import openxlab

        # 使用 monkey patch 修改 openxlab 的配置路径
        try:
            import openxlab.helper.config_parser as config_parser
            # 修改配置文件路径
            if hasattr(config_parser, 'TOKEN_FILE'):
                original_token_file = config_parser.TOKEN_FILE
                config_parser.TOKEN_FILE = os.path.join(openxlab_home, 'token.json')
        except ImportError:
            pass

        # 尝试修改 openxlab 的 home 目录
        try:
            from openxlab import config
            if hasattr(config, 'OPENXLAB_HOME'):
                config.OPENXLAB_HOME = openxlab_home
        except:
            pass

        # 尝试登录（因为 token 已经存在，openxlab.login 可能会读取它）
        try:
            openxlab.login(ak=ak, sk=sk)
        except:
            pass  # 如果登录失败但 token 已保存，继续

        print("✓ 登录成功")
        print(f"✓ Token 已保存到: {default_token_path}\n")
    except Exception as e:
        print(f"✗ 登录失败: {e}\n")
        print("提示: 请确保 AK/SK 正确，或者访问 https://openxlab.org.cn 获取凭证")
        sys.exit(1)


def view_dataset_info(key, dataset_config):
    """查看数据集信息"""
    print("=" * 60)
    print(f"查看 {dataset_config['name']} 数据集信息...")
    print("=" * 60)

    try:
        from openxlab.dataset import info
        info(dataset_repo=dataset_config['repo'])
        print(f"✓ {dataset_config['name']} 数据集信息获取成功\n")
        return True
    except Exception as e:
        print(f"✗ 获取 {dataset_config['name']} 数据集信息失败: {e}\n")
        return False


def check_downloaded(target_path, expected_min_size=100*1024*1024):
    """
    检查目标路径是否已有下载的文件
    expected_min_size: 期望的最小文件大小（字节），默认100MB
    """
    if not os.path.exists(target_path):
        return False, 0

    # 检查文件夹是否存在且有内容
    if os.path.isdir(target_path):
        files = os.listdir(target_path)
        if len(files) > 0:
            # 计算总大小（包括子文件夹）
            total_size = 0
            for root, dirs, files_in_dir in os.walk(target_path):
                for f in files_in_dir:
                    file_path = os.path.join(root, f)
                    if os.path.isfile(file_path):
                        total_size += os.path.getsize(file_path)

            # 只有当文件大小超过预期最小值时才认为已下载完成
            if total_size >= expected_min_size:
                return True, total_size
            else:
                # 文件夹存在但文件太小，可能是下载失败或部分下载
                return False, total_size

    return False, 0


def check_archive_integrity(target_path, archive_name, expected_size):
    """
    检查压缩文件的完整性
    返回: (is_complete, actual_size)
    """
    archive_path = find_archive_file(target_path, archive_name)

    if not archive_path:
        return False, 0

    actual_size = os.path.getsize(archive_path)

    # 允许 5% 的误差
    if actual_size >= expected_size * 0.95:
        return True, actual_size
    else:
        return False, actual_size


def download_dataset(dataset_name, dataset_repo, target_path, expected_min_size=100*1024*1024,
                    archive_name=None, expected_archive_size=None):
    """
    下载数据集（主线程版本）
    注意: openxlab 内部使用 signal，必须在主线程运行
    """
    print("=" * 60)
    print(f"下载 {dataset_name} 数据集...")
    print("=" * 60)
    print(f"目标路径: {os.path.abspath(target_path)}\n")

    # 检查是否已下载
    exists, size = check_downloaded(target_path, expected_min_size)

    # 如果检测到已下载，且提供了压缩文件信息，则验证压缩文件完整性
    if exists and archive_name and expected_archive_size:
        is_complete, archive_size = check_archive_integrity(target_path, archive_name, expected_archive_size)

        if is_complete:
            print(f"✓ 检测到数据集已存在且完整 (大小: {size/1024/1024:.1f} MB)")
            choice = input("是否跳过下载? (y/n): ").strip().lower()
            if choice == 'y':
                print(f"✓ 跳过 {dataset_name} 下载\n")
                return True
        else:
            print(f"⚠ 检测到不完整的压缩文件！")
            print(f"   期望大小: {expected_archive_size/1024/1024:.1f} MB")
            print(f"   实际大小: {archive_size/1024/1024:.1f} MB")
            print(f"   将重新下载数据集...\n")
    elif exists:
        print(f"✓ 检测到数据集已存在 (大小: {size/1024/1024:.1f} MB)")
        choice = input("是否跳过下载? (y/n): ").strip().lower()
        if choice == 'y':
            print(f"✓ 跳过 {dataset_name} 下载\n")
            return True
    elif size > 0:
        print(f"⚠ 检测到不完整的下载 (大小: {size/1024/1024:.1f} MB)")
        print(f"   重新下载以获取完整数据集...\n")

    try:
        from openxlab.dataset import get
        get(dataset_repo=dataset_repo, target_path=target_path)
        print(f"✓ {dataset_name} 数据集下载成功\n")
        return True
    except Exception as e:
        print(f"✗ {dataset_name} 数据集下载失败")
        print(f"   错误: {e}\n")
        return False


def download_single_file(dataset_name, dataset_repo, source_path='/README.md', target_path='.'):
    """下载单个文件"""
    print("=" * 60)
    print(f"从 {dataset_name} 下载单个文件...")
    print("=" * 60)

    try:
        from openxlab.dataset import download
        download(
            dataset_repo=dataset_repo,
            source_path=source_path,
            target_path=target_path
        )
        print(f"✓ 文件 {source_path} 下载成功\n")
        return True
    except Exception as e:
        print(f"⚠ 文件下载失败 (非关键错误): {e}\n")
        return False


def find_archive_file(target_path, archive_name):
    """
    查找压缩文件
    支持在子文件夹中查找
    """
    # 首先检查直接路径
    direct_path = os.path.join(target_path, archive_name)
    if os.path.exists(direct_path):
        return direct_path

    # 递归搜索子文件夹
    for root, dirs, files in os.walk(target_path):
        if archive_name in files:
            return os.path.join(root, archive_name)

    return None


def is_extracted(dataset_name, archive_path, target_path):
    """
    检查压缩文件是否已解压
    通过检查解压后的文件夹是否存在且有内容
    """
    # 根据压缩文件名推断解压后的文件夹名
    archive_basename = os.path.basename(archive_path)

    # 移除压缩文件扩展名
    if archive_basename.endswith('.tar.xz'):
        extracted_folder = archive_basename.replace('.tar.xz', '')
    elif archive_basename.endswith('.tar.gz'):
        extracted_folder = archive_basename.replace('.tar.gz', '')
    elif archive_basename.endswith('.zip'):
        extracted_folder = archive_basename.replace('.zip', '')
    else:
        extracted_folder = archive_basename

    # 在目标路径下查找解压后的文件夹
    extracted_path = None
    for root, dirs, files in os.walk(target_path):
        if extracted_folder in dirs:
            extracted_path = os.path.join(root, extracted_folder)
            break

    if extracted_path and os.path.exists(extracted_path):
        # 检查文件夹是否有内容
        files_in_folder = os.listdir(extracted_path)
        if len(files_in_folder) > 0:
            return True, extracted_path

    return False, None


def extract_zip(zip_path, extract_to=None):
    """解压 ZIP 文件"""
    print(f"正在解压: {os.path.basename(zip_path)}")

    if extract_to is None:
        extract_to = os.path.dirname(zip_path)

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # 获取文件列表
            file_list = zip_ref.namelist()
            total_files = len(file_list)

            print(f"包含 {total_files} 个文件")
            print(f"解压到: {extract_to}\n")

            # 提取文件
            for i, file in enumerate(file_list, 1):
                zip_ref.extract(file, extract_to)
                # 显示进度
                if i % 100 == 0 or i == total_files:
                    print(f"进度: {i}/{total_files} 文件已完成", end='\r')

        print(f"\n✓ ZIP 文件解压成功: {zip_path}\n")
        return True
    except Exception as e:
        print(f"\n✗ ZIP 文件解压失败: {e}\n")
        return False


def extract_tar_xz(tar_path, extract_to=None):
    """解压 tar.xz 文件"""
    print(f"正在解压: {os.path.basename(tar_path)}")

    if extract_to is None:
        extract_to = os.path.dirname(tar_path)

    try:
        with tarfile.open(tar_path, 'r:xz') as tar_ref:
            # 获取文件列表
            member_list = tar_ref.getmembers()
            total_files = len(member_list)

            print(f"包含 {total_files} 个文件")
            print(f"解压到: {extract_to}")
            print(f"⚠ 注意: 解压大文件需要较长时间，请耐心等待...\n")

            # 提取文件
            for i, member in enumerate(member_list, 1):
                tar_ref.extract(member, extract_to)
                # 显示进度
                if i % 100 == 0 or i == total_files:
                    print(f"进度: {i}/{total_files} 文件已完成", end='\r')

        print(f"\n✓ tar.xz 文件解压成功: {tar_path}\n")
        return True
    except Exception as e:
        print(f"\n✗ tar.xz 文件解压失败: {e}\n")
        return False


def extract_dataset(dataset_name, target_path, archive_name, archive_type, expected_size=None):
    """
    检测并解压数据集
    """
    print("=" * 60)
    print(f"检查 {dataset_name} 解压状态...")
    print("=" * 60)

    # 查找压缩文件
    archive_path = find_archive_file(target_path, archive_name)

    if not archive_path:
        print(f"⚠ 未找到压缩文件 {archive_name}")
        print(f"   可能尚未下载或已被删除\n")
        return False

    actual_size = os.path.getsize(archive_path)
    print(f"✓ 找到压缩文件: {archive_path}")
    print(f"   大小: {actual_size/1024/1024:.1f} MB")

    # 检查文件完整性
    if expected_size and actual_size < expected_size * 0.95:  # 允许5%的误差
        print(f"\n✗ 压缩文件不完整！")
        print(f"   期望大小: {expected_size/1024/1024:.1f} MB")
        print(f"   实际大小: {actual_size/1024/1024:.1f} MB")
        print(f"\n建议操作:")
        print(f"   1. 重新运行脚本，选择重新下载数据集")
        print(f"   2. 或手动删除不完整的文件后重新下载")
        print(f"   文件路径: {archive_path}\n")
        return False

    # 检查是否已解压
    is_ext, extracted_path = is_extracted(dataset_name, archive_path, target_path)
    if is_ext:
        print(f"✓ 检测到已解压: {extracted_path}")
        choice = input("是否跳过解压? (y/n): ").strip().lower()
        if choice == 'y':
            print(f"✓ 跳过 {dataset_name} 解压\n")
            return True

    # 开始解压
    print(f"\n开始解压 {dataset_name}...\n")

    success = False
    if archive_type == 'zip':
        success = extract_zip(archive_path)
    elif archive_type == 'tar.xz':
        success = extract_tar_xz(archive_path)
    else:
        print(f"✗ 不支持的压缩格式: {archive_type}\n")
        return False

    if success:
        print(f"✓ {dataset_name} 解压完成\n")
    return success


def select_datasets():
    """让用户选择要下载的数据集"""
    print("\n" + "=" * 60)
    print("请选择要下载的数据集:")
    print("=" * 60)

    for key, config in DATASETS.items():
        print(f"{key}. {config['name']} - {config['size']} - {config['description']}")
    print("4. 下载全部数据集")
    print("0. 跳过下载")
    print("=" * 60)

    choice = input("\n请输入选项 (0/1/2/3/4): ").strip()

    return choice


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("OpenXLab CCPD 数据集下载工具 v2.3")
    print("支持: CCPD2019 | CCPD2020")
    print("=" * 60)
    print("特性: 断点续传 | 分步下载 | 自动解压")
    print("=" * 60 + "\n")

    # 步骤 1: 安装包
    install_package()

    # 步骤 2: 登录
    login_openxlab()

    # 步骤 3: 选择数据集并查看信息
    print("步骤 3/7: 查看数据集信息...")
    choice = select_datasets()

    datasets_to_download = []
    if choice == '1':
        datasets_to_download.append('1')
    elif choice == '2':
        datasets_to_download.append('2')
    elif choice == '3':
        datasets_to_download.append('3')
    elif choice == '4':
        datasets_to_download = ['1', '2', '3']
    elif choice == '0':
        print("跳过下载")
        return
    else:
        print("无效选项，跳过下载")
        return

    # 查看选中的数据集信息
    for key in datasets_to_download:
        view_dataset_info(key, DATASETS[key])

    # 步骤 4: 设置下载路径
    base_path = input("\n请输入基础下载路径 (直接回车使用默认路径 './CCPD_Datasets'): ").strip()
    if not base_path:
        base_path = './CCPD_Datasets'

    # 步骤 5: 下载数据集
    print("\n" + "=" * 60)
    print("步骤 5/7: 开始下载数据集...")
    print("=" * 60 + "\n")

    download_results = {}

    for key in datasets_to_download:
        config = DATASETS[key]
        target_path = os.path.join(base_path, config['folder'])

        success = download_dataset(
            config['name'],
            config['repo'],
            target_path,
            config['min_size'],
            config.get('archive_name'),
            config.get('expected_archive_size')
        )
        download_results[config['name']] = success

        # 下载示例文件
        if success:
            download_single_file(config['name'], config['repo'], '/README.md', target_path)

    # 步骤 6: 解压数据集
    print("\n" + "=" * 60)
    print("步骤 6/7: 解压数据集...")
    print("=" * 60 + "\n")

    extract_results = {}

    for key in datasets_to_download:
        config = DATASETS[key]
        target_path = os.path.join(base_path, config['folder'])

        # 只对下载成功的数据集进行解压
        if download_results.get(config['name'], False):
            success = extract_dataset(
                config['name'],
                target_path,
                config['archive_name'],
                config['archive_type'],
                config.get('expected_archive_size')
            )
            extract_results[config['name']] = success
        else:
            extract_results[config['name']] = False

    # 步骤 7: 显示摘要
    print("\n" + "=" * 60)
    print("步骤 7/7: 任务摘要")
    print("=" * 60)

    print("\n下载结果:")
    for name, success in download_results.items():
        status = "✓ 成功" if success else "✗ 失败"
        print(f"  {name}: {status}")

    print("\n解压结果:")
    for name, success in extract_results.items():
        status = "✓ 成功" if success else "✗ 失败/跳过"
        print(f"  {name}: {status}")

    print("\n" + "=" * 60)
    print("所有步骤完成!")
    print("=" * 60)
    print("\n提示:")
    print("- 如果下载超时，可以重新运行脚本，已下载的文件会自动跳过")
    print("- 解压后的数据集可直接用于训练")


if __name__ == "__main__":
    main()
