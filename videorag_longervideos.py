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
    # multiprocessing.set_start_method('spawn')
    multiprocessing.set_start_method('spawn', force=True)
    ## learn
    video_base_path = f'/data/huangjie/Project/upProject/VideoAgent_api/videos/origin'
    video_files = sorted(os.listdir(video_base_path))
    video_paths = [os.path.join(video_base_path, f) for f in video_files]
    videorag = VideoRAG( working_dir=f"/data/huangjie/Project/upProject/VideoAgent_api/working_dir")    
    videorag.insert_video(video_path_list=video_paths)
 
    querys = "在911这个视频中，什么时候出现三个人"
    response = videorag.query(query=querys, param=QueryParam)
    print("ans:\n\n", response)
