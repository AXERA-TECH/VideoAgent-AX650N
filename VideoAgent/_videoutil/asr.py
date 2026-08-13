

import os
import torch
import logging
from tqdm import tqdm

# from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import os
from .._llm import ASRModel, SherpaOnnxASRClient

model = ASRModel()
OnnxASRClient = SherpaOnnxASRClient()
def speech_to_text(video_name, working_dir, segment_index2name, audio_output_format):
    # model_path = os.getenv("WhisperModel_Path")
    # model = WhisperModel(model_path)
    # model.logger.setLevel(logging.WARNING)

    cache_path = os.path.join(working_dir, '_cache', video_name)

    transcripts = {}
    for index in tqdm(segment_index2name, desc=f"Speech Recognition {video_name}"):
        segment_name = segment_index2name[index]
        audio_file = os.path.join(cache_path, f"{segment_name}.{audio_output_format}")

        # if the audio file does not exist, skip it
        if not os.path.exists(audio_file):
            transcripts[index] = ""
            continue

        # result = model.transcribe(audio_file)
        result = OnnxASRClient.transcribe(audio_file)
        # 处理不同的返回类型
        if isinstance(result, tuple):
            # 如果返回 tuple，提取文本
            text_content = result[0] if result else ""
        elif hasattr(result, 'text'):
            # 如果是对象，获取 text 属性
            text_content = result.text
        else:
            # 其他情况，转换为字符串
            text_content = str(result)

        # print("Transcription:, ", text_content)
        transcripts[index] = text_content

    return transcripts
