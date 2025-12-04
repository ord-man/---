"""
作者：王艺 
学校：sau
语音
"""
import shutil
import subprocess
import sys

def check_ffmpeg():
    print("🔍 正在检查 ffmpeg 是否安装并配置到系统 PATH...")

    # 方式一：使用 shutil 检测可执行路径
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        print("❌ 未找到 ffmpeg。请确认你已安装并将其添加到系统环境变量 PATH 中。")
        print("👉 建议操作：")
        print("  1. 下载地址：https://www.gyan.dev/ffmpeg/builds/")
        print("  2. 解压并添加 C:\\ffmpeg\\bin 到环境变量 PATH")
        print("  3. 重启终端或编辑器")
        sys.exit(1)
    else:
        print(f"✅ 找到 ffmpeg 可执行文件：{ffmpeg_path}")

    # 方式二：尝试运行 ffmpeg -version
    try:
        output = subprocess.check_output(['ffmpeg', '-version'], stderr=subprocess.STDOUT, text=True)
        print("✅ ffmpeg 已正确运行。版本信息如下：\n")
        print(output.splitlines()[0])  # 只显示首行版本号
    except subprocess.CalledProcessError as e:
        print("⚠️ 运行 ffmpeg 时出错：")
        print(e.output)
    except Exception as e:
        print("❌ 执行 ffmpeg 失败：", str(e))
        sys.exit(1)

if __name__ == "__main__":
    check_ffmpeg()

