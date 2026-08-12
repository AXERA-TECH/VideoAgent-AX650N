import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv
from moviepy.video.io.VideoFileClip import VideoFileClip
load_dotenv()
from .._llm import Qwen3VLModel

def encode_video(video, frame_times):
    frames = []
    for t in frame_times:
        frames.append(video.get_frame(t))
    frames = np.stack(frames, axis=0)
    frames = [Image.fromarray(v.astype('uint8')).resize((640, 360)) for v in frames]
    return frames

model = Qwen3VLModel()
def segment_caption(video_name, video_path, segment_index2name, transcripts, segment_times_info, caption_result, error_queue):
    try:

        with VideoFileClip(video_path) as video:
            for index in tqdm(segment_index2name, desc=f"Captioning Video {video_name}"):
                frame_times = segment_times_info[index]["frame_times"]
                video_frames = encode_video(video, frame_times)
                segment_transcript = transcripts[index]
                query = f"The transcript of the current video:\n{segment_transcript}.\nNow provide a description (caption) of the video in English."
               
                messages_input = [
 
                    {"role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": video_frames,
                        },
                        {"type": "text", "text": query},
                    ],
                }
                    
                ]

                segment_caption = model.generate_result(messages_input, max_new_tokens=256)
                caption_result[index] = segment_caption
                del video_frames
    except Exception as e:
        error_queue.put(f"Error in segment_caption:\n {str(e)}")
        raise RuntimeError

def merge_segment_information(segment_index2name, segment_times_info, transcripts, captions):
    inserting_segments = {}
    for index in segment_index2name:
        inserting_segments[index] = {"content": None, "time": None}
        segment_name = segment_index2name[index]
        inserting_segments[index]["time"] = '-'.join(segment_name.split('-')[-2:])
        
        # 检查caption是否存在，如果不存在则使用transcript作为fallback
        if index in captions:
            caption_text = captions[index]
        else:
            caption_text = f"[Caption generation failed for segment {index}]"
        
        inserting_segments[index]["content"] = f"Caption:\n{caption_text}\nTranscript:\n{transcripts[index]}\n\n"
        inserting_segments[index]["transcript"] = transcripts[index]
        inserting_segments[index]["frame_times"] = segment_times_info[index]["frame_times"].tolist()
    return inserting_segments
        

def retrieved_segment_caption(retrieved_segments, video_path_db, video_segments, num_sampled_frames):
    caption_result = {}
    for this_segment in tqdm(retrieved_segments, desc='Captioning Segments for Given Query'):
        video_name = '_'.join(this_segment.split('_')[:-1])
        index = this_segment.split('_')[-1]
        video_path = video_path_db._data[video_name]
        timestamp = video_segments._data[video_name][index]["time"].split('-')
        start, end = eval(timestamp[0]), eval(timestamp[1])
        video = VideoFileClip(video_path)
        frame_times = np.linspace(start, end, num_sampled_frames, endpoint=False)
        
        video_frames = encode_video(video, frame_times)
        segment_transcript = video_segments._data[video_name][index]["transcript"]
        
        query = f"The transcript of the current video:\n{segment_transcript}.\nNow provide a very detailed description (caption) of the video in English."
        
        messages_input = [
 
                    {"role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": video_frames,
                        },
                        {"type": "text", "text": query},
                    ],
                }
                    
                ]

        segment_caption = model.generate_result(messages_input, max_new_tokens=256)

        caption_result[this_segment] = f"Caption:\n{segment_caption}\nTranscript:\n{segment_transcript}\n\n"

        del video_frames
        video.close()
    
    return caption_result