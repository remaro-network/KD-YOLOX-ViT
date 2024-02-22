import os
import loguru as logger
import subprocess
import glob

folder_path = 'datasets/COCO/SWDD/train2017/'
pattern = '*.png'

for filepath in glob.glob(os.path.join(folder_path, pattern)):
    command = [
        "python3", 
        "tools/demo.py", 
        "image", 
        "-n", "yolox-l", 
        "--path", "datasets/COCO/SWDD/train2017/", 
        "--conf", "0.5", 
        "--nms", "0.45", 
        "--tsize", "416",#"640",#"416", 
        "--ckpt", "YOLOX_outputs/yolox_l/epoch_300_ckpt.pth", 
        "--save_result", 
        "--device", "gpu"
    ]
    subprocess.run(command)

