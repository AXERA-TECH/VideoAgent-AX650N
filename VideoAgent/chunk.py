import logging
logger = logging.getLogger("nano-graphrag")
# from dashscope import get_tokenizer 
from transformers import AutoTokenizer
from ._utils import compute_mdhash_id
from typing import Callable, Dict, List, Optional, Type, Union, cast
import asyncio
import os
from ._llm import Qwen3
from ._llm import Qwen3TokenizerClient


qwen3_model = Qwen3()
tiktoken_client = Qwen3TokenizerClient()


def chunking_by_video_segments(
    tokens_list: list[list[int]],
    doc_keys,
    max_token_size=1024,
):
    # make sure each segment is not larger than max_token_size
    for index in range(len(tokens_list)):
        if len(tokens_list[index]) > max_token_size:
            tokens_list[index] = tokens_list[index][:max_token_size]
    
    results = []
    chunk_token = []
    chunk_segment_ids = []
    chunk_order_index = 0
    for index, tokens in enumerate(tokens_list):
        
        if len(chunk_token) + len(tokens) <= max_token_size:
            # add new segment
            chunk_token += tokens.copy()
            chunk_segment_ids.append(doc_keys[index])
        else:
            

            chunk = tiktoken_client.decode(chunk_token, skip_special_tokens=True)
            results.append(
                {
                    "tokens": len(chunk_token),
                    "content": chunk["text"],
                    "chunk_order_index": chunk_order_index,
                    "video_segment_id": chunk_segment_ids,
                }
            )
            # new chunk with current segment as begin
            chunk_token = []
            chunk_segment_ids = []
            chunk_token += tokens.copy()
            chunk_segment_ids.append(doc_keys[index])
            chunk_order_index += 1
    
    # save the last chunk
    if len(chunk_token) > 0:

        chunk = tiktoken_client.decode(chunk_token, skip_special_tokens=True)
        results.append(
            {
                "tokens": len(chunk_token),
                "content": chunk["text"],
                "chunk_order_index": chunk_order_index,
                "video_segment_id": chunk_segment_ids,
            }
        )
    
    return results


def get_chunks(new_videos, chunk_func=chunking_by_video_segments, **chunk_func_params):
    inserting_chunks = {}

    new_videos_list = list(new_videos.keys())
    for video_name in new_videos_list:
        segment_id_list = list(new_videos[video_name].keys())
        docs = [new_videos[video_name][index]["content"] for index in segment_id_list]
        doc_keys = [f'{video_name}_{index}' for index in segment_id_list]


        tokens_dict = tiktoken_client.batch_encode(docs, padding=True)
        tokens_list = tokens_dict.get("token_ids_batch", tokens_dict) if isinstance(tokens_dict, dict) else tokens_dict

        chunks = chunk_func(
            tokens_list, doc_keys=doc_keys,  **chunk_func_params
        )

        for chunk in chunks:
            inserting_chunks.update(
                {compute_mdhash_id(chunk["content"], prefix="chunk-"): chunk}
            )

    return inserting_chunks





