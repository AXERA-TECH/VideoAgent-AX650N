import os
import json
import logging
import warnings
import multiprocessing
import sys
from dotenv import load_dotenv
load_dotenv()

warnings.filterwarnings("ignore")
logging.getLogger("httpx").setLevel(logging.WARNING)



from VideoAgent import VideoRAG, QueryParam



if __name__ == '__main__':

    multiprocessing.set_start_method('spawn', force=True)

    # video文件地址:
    video_base_path = f'/root/huangjie/VideoAgent_api513/videos/origin'
    video_files = sorted(os.listdir(video_base_path))
    video_paths = [os.path.join(video_base_path, f) for f in video_files]
    
    #工作目录
    videorag = VideoRAG( working_dir=f"/root/huangjie/VideoAgent_api513/working_dir")    
    videorag.insert_video(video_path_list=video_paths)
 
    querys = "SP视频开头前10秒的内容"
    response = videorag.query(query=querys, param=QueryParam)
    print("ans:\n\n", response)
