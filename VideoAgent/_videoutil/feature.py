import os
import numpy as np
import torch

from PIL import Image


from .._llm import Qwen3VLEmbedderC

from .._utils import _pil_to_base64

import os
import torch
import unicodedata
import numpy as np
from PIL import Image
from urllib.parse import urlparse
from typing import Optional, List, Union, Dict, Any

# Constants for configuration
IMAGE_BASE_FACTOR = 16
IMAGE_FACTOR = IMAGE_BASE_FACTOR * 2
MIN_PIXELS = 4 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_PIXELS = 1800 * IMAGE_FACTOR * IMAGE_FACTOR
FPS = 1
MAX_FRAMES = 10
FRAME_MAX_PIXELS = 768 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_TOTAL_PIXELS = 10 * FRAME_MAX_PIXELS


_MAX_FRAMES = 5

model = None


def _get_model():
    global model
    if model is None:
        model = Qwen3VLEmbedderC()
    return model

def sample_frames(frames: List[Union[str, Image.Image]], max_segments: int) -> List[Union[str, Image.Image]]:
    duration = len(frames)
    if duration <= max_segments:
        return frames

    frame_id_array = np.linspace(0, duration - 1, max_segments, dtype=int)
    frame_id_list = frame_id_array.tolist()
    sampled_frames = [ frames[frame_idx] for frame_idx in frame_id_list ]
    return sampled_frames

def is_image_path(path: str) -> bool:
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.svg'}
    
    if path.startswith(('http://', 'https://')):
        # Parse URL to remove query parameters
        parsed_url = urlparse(path)
        clean_path = parsed_url.path
    else:
        clean_path = path
    
    # Check file extension
    _, ext = os.path.splitext(clean_path.lower())
    return ext in image_extensions

def is_video_input(video) -> bool:
    if isinstance(video, str):
        return True
    
    if isinstance(video, list) and len(video) > 0:
        # Check first element to determine the type
        first_elem = video[0]
        
        if isinstance(first_elem, Image.Image):
            return True
        
        if isinstance(first_elem, str):
            return is_image_path(first_elem)
    
    return False

def format_model_input(
        text: Optional[Union[List[str], str]] = None,
        image: Optional[Union[List[Union[str, Image.Image]], str, Image.Image]] = None,
        video: Optional[Union[List[Union[str, List[Union[str, Image.Image]]]], str, List[Union[str, Image.Image]]]] = None,
        instruction: Optional[str] = None,
        fps: Optional[float] = None,
        max_frames: Optional[int] = None,
        default_instruction: str = "Represent the user's input.",
        min_pixels: int = MIN_PIXELS,
        max_pixels: int = MAX_PIXELS,
        total_pixels: int = MAX_TOTAL_PIXELS,
        max_frame_num: int = MAX_FRAMES
    ) -> List[Dict]:

    # Ensure instruction ends with punctuation
    if instruction:
        instruction = instruction.strip()
        if instruction and not unicodedata.category(instruction[-1]).startswith('P'):
            instruction = instruction + '.'

    # Initialize conversation with system prompts
    content = []
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": instruction or default_instruction}]},
        {"role": "user", "content": content}
    ]

    # Normalize text input to list
    if text is None:
        texts = []
    elif isinstance(text, str):
        texts = [text]
    else:
        texts = text
    
    # Normalize image input to list
    if image is None:
        images = []
    elif not isinstance(image, list):
        images = [image]
    else:
        images = image
    
    # Normalize video input to list
    if video is None:
        videos = []
    elif is_video_input(video):
        videos = [video]
    else:
        # Assume it's a list of videos
        videos = video

    # Add text, image, or video content to conversation
    if not texts and not images and not videos:
        content.append({'type': 'text', 'text': "NULL"})
        return conversation

    # Process each video
    for vid in videos:
        video_content = None
        video_kwargs = {'total_pixels': total_pixels}
        
        if isinstance(vid, list):
            # Video as frame sequence
            video_content = vid
            if max_frame_num is not None:
                video_content = sample_frames(video_content, max_frame_num)
            video_content = [
                ('file://' + ele if isinstance(ele, str) else ele) 
                for ele in video_content
            ]
            print("video_content:", video_content)
        elif isinstance(vid, str):
            # Video as file path
            video_content = vid if vid.startswith(('http://', 'https://')) else 'file://' + vid
            video_kwargs = {'fps': fps or FPS, 'max_frames': max_frames or max_frame_num}
        else:
            raise TypeError(f"Unrecognized video type: {type(vid)}")

        # Add video input to content
        if video_content:
            content.append({
                'type': 'video', 
                'video': video_content,
                **video_kwargs
            })

    # Process each image
    for img in images:
        image_content = None
        
        if isinstance(img, Image.Image):
            image_content = img
        elif isinstance(img, str):
            image_content = img if img.startswith(('http://', 'https://')) else 'file://' + img
        else:
            raise TypeError(f"Unrecognized image type: {type(img)}")

        # Add image input to content
        if image_content:
            content.append({
                'type': 'image', 
                'image': image_content,
                "min_pixels": min_pixels,
                "max_pixels": max_pixels
            })

    # Process each text
    for txt in texts:
        content.append({'type': 'text', 'text': txt})

    return conversation


def _load_frames(video_path: str, max_frames: int = _MAX_FRAMES):
    """用 OpenCV 均匀采样至多 max_frames 帧，返回 PIL Image 列表。"""
    import cv2

    cap = cv2.VideoCapture(video_path)
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            raise ValueError(f"Cannot read video: {video_path}")
        indices = np.linspace(0, total - 1, min(total, max_frames), dtype=int)
        frames = []
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
    finally:
        cap.release()
    return frames



def encode_video_segments(video_path):
    """Load frames locally, send as base64 images to the remote embedding API."""
    embedder = _get_model()
    frames = _load_frames(video_path)
    encoded_frames = [_pil_to_base64(frame) for frame in frames]

    # Build conversation compatible with embedding_server.py → process_vision_info → fetch_image
    # Use type:"image" with data:image base64 string directly (avoid image_url dict nesting)
    image_items = []
    for b64 in encoded_frames:
        image_items.append({
            "type": "image",
            "image": f"data:image/jpeg;base64,{b64}",
        })

    conversation = [
        {"role": "system", "content": [{"type": "text", "text": "Represent the user's input."}]},
        {"role": "user", "content": image_items},
    ]

    embeddings = embedder.embedding_gen(conversation)
    tensor_embeddings = torch.tensor(embeddings, dtype=torch.float32).unsqueeze(0)
    return tensor_embeddings


def encode_image(image_path: str):
    """编码文本查询为嵌入向量"""
    embedder = _get_model()
    image = Image.open(image_path)

    encoded_frame = _pil_to_base64(image)
    content = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_frame}"}}
    embeddings = embedder.video_embedding(content=content)

    tensor_embeddings = torch.tensor(embeddings, dtype=torch.float32).unsqueeze(0)
    return tensor_embeddings


def encode_string_query(query: str):
    """编码文本查询为嵌入向量"""
    embedder = _get_model()

    message_input = format_model_input(text=query)
    embeddings = embedder.embedding_gen(message_input)
    tensor_embeddings = torch.tensor(embeddings, dtype=torch.float32).unsqueeze(0)

    return tensor_embeddings