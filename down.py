"""
作者：王艺 
学校：sau
"""
import os, zipfile, urllib.request

# 下载地址（官方源）
url = "https://alphacephei.com/vosk/models/vosk-model-small-cn-0.22.zip"
target_zip = r"E:\asr\vosk-cn-small.zip"
extract_dir = r"E:\asr\vosk-cn-small"

os.makedirs(os.path.dirname(target_zip), exist_ok=True)

print(f"Downloading {url} -> {target_zip} ...")
urllib.request.urlretrieve(url, target_zip)
print("Download finished.")

print(f"Extracting to {extract_dir} ...")
with zipfile.ZipFile(target_zip, 'r') as zip_ref:
    zip_ref.extractall(os.path.dirname(extract_dir))

# 解压后一般是 vosk-model-small-cn-0.22 文件夹，把它重命名为 vosk-cn-small
extracted_root = os.path.join(os.path.dirname(extract_dir), "vosk-model-small-cn-0.22")
if os.path.exists(extracted_root):
    if os.path.exists(extract_dir):
        print(f"Warning: {extract_dir} already exists, skip renaming")
    else:
        os.rename(extracted_root, extract_dir)

print("Done. Model ready at:", extract_dir)
