import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv
from moviepy.video.io.VideoFileClip import VideoFileClip
load_dotenv()
from .._llm import Qwen3VLModel
from .._utils import _pil_to_base64
import logging

# 将日志级别设置为 WARNING，这样 INFO 级别的日志就会被忽略
logging.getLogger("httpx").setLevel(logging.WARNING)
def encode_video(video, frame_times):
    frames = []
    for t in frame_times:
        frames.append(video.get_frame(t))
    frames = np.stack(frames, axis=0)
    frames = [Image.fromarray(v.astype('uint8')) for v in frames]
    return frames

model = Qwen3VLModel()
def segment_caption(video_name, video_path, segment_index2name, transcripts, segment_times_info, caption_result, error_queue):
    try:

        with VideoFileClip(video_path) as video:
            for index in tqdm(segment_index2name, desc=f"Captioning Video {video_name}"):
                frame_times = segment_times_info[index]["frame_times"]
                video_frames = encode_video(video, frame_times)
                segment_transcript = transcripts[index]
                query = f"The transcript of the current video:\n{segment_transcript}.\nNow provide a description (caption) of the video in Chinese."
                
                encoded_frames = [_pil_to_base64(frame) for frame in video_frames]
        
                content = []
                for encoded_frame in encoded_frames:
                    content.append( {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_frame}"}})

                content.append({"type": "text", "text": query})

                messages_input = [

                    {"role": "user",
                    "content": content
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
        
        query = f"The transcript of the current video:\n{segment_transcript}.\nNow provide a very detailed description (caption) of the video in Chinese."
        
        encoded_frames = [_pil_to_base64(frame) for frame in video_frames]
        
        content = []
        for encoded_frame in encoded_frames:
            content.append( {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_frame}"}})

        content.append({"type": "text", "text": query})

        messages_input = [

            {"role": "user",
            "content": content
        }
            
        ]

        segment_caption = model.generate_result(messages_input, max_new_tokens=256)

        caption_result[this_segment] = f"Caption:\n{segment_caption}\nTranscript:\n{segment_transcript}\n\n"

        del video_frames
        video.close()
    
    return caption_result


def retrieved_segment_caption_kw(retrieved_segments, video_path_db, video_segments, refine_knowledge, num_sampled_frames):
    caption_result = {}
    for this_segment in tqdm(retrieved_segments, desc='Captioning Segments for Given Query'):
        video_name = '_'.join(this_segment.split('_')[:-1])
        index = this_segment.split('_')[-1]
        video_path = video_path_db._data[video_name]

        # Check if video file exists before trying to open it
        if not os.path.exists(video_path):
            error_msg = f"Video file not found for segment '{this_segment}': {video_path}"
            raise FileNotFoundError(error_msg)

        try:
            timestamp = video_segments._data[video_name][index]["time"].split('-')
            start, end = eval(timestamp[0]), eval(timestamp[1])
            video = VideoFileClip(video_path)
            frame_times = np.linspace(start, end, num_sampled_frames, endpoint=False)

            video_frames = encode_video(video, frame_times)
            segment_transcript = video_segments._data[video_name][index]["transcript"]

            query = f"The transcript of the current video:\n{segment_transcript}.\nNow provide a very detailed description (caption) of the video in Chinese and extract relevant information about: {refine_knowledge}"

            encoded_frames = [_pil_to_base64(frame) for frame in video_frames]
            # print("encoded_frames: \n",len(encoded_frames))
            content = []
            for encoded_frame in encoded_frames:
                content.append( {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_frame}"}})

            content.append({"type": "text", "text": query})

            messages_input = [

                {"role": "user",
                "content": content
            }
                
            ]

            segment_caption = model.generate_result(messages_input, max_new_tokens=256)

            caption_result[this_segment] = f"Caption:\n{segment_caption}\nTranscript:\n{segment_transcript}\n\n"

            del video_frames
            video.close()
        except FileNotFoundError:
            raise
        except Exception as e:
            error_msg = f"Error processing segment '{this_segment}' (video: {video_path}): {str(e)}"
            raise RuntimeError(error_msg) from e

    return caption_result