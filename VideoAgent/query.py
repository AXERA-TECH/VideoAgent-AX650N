import os
import numbers
import asyncio
from typing import Any, Union
from .prompt import PROMPTS

from ._utils import logger, clean_output
from ._llm import Qwen3
from ._llm import Qwen3TokenizerClient
from ._videoutil import (
   
    retrieved_segment_caption,
    retrieved_segment_caption_kw,
)

from dotenv import load_dotenv
from transformers import AutoTokenizer
import re

load_dotenv()




qwen3_model = Qwen3()

tiktoken_client = Qwen3TokenizerClient()



def truncate_list_by_token_size(list_data: list, key: callable, max_token_size: int):
    """Truncate a list of data by token size"""
    if max_token_size <= 0:
        return []
    tokens = 0
    for i, data in enumerate(list_data):
        # tokens += len(tiktoken_model_p.encode(key(data)))

        tokens += len(tiktoken_client.encode(text=key(data)).get("token_ids", []))
        if tokens > max_token_size:
            return list_data[:i]
    return list_data

def _extract_keywords_query(
    query
):
    # use_llm_func: callable = global_config["llm"]["cheap_model_func"]
    keywords_prompt = PROMPTS["keywords_extraction"]
    keywords_prompt = keywords_prompt.format(input_text=query)
    messages = [
        {"role": "user", "content": keywords_prompt}
    ]
    final_result = qwen3_model.generate_result(messages, max_new_tokens=256)
    return final_result

def _result_query_stream(query, sys_prompt, max_new_tokens: int = 1024):
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": query}
    ]
    return qwen3_model.generate_result_stream(messages, max_new_tokens=max_new_tokens)

def _result_query_back(query,sys_prompt):

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": query}
    ]
 
    # content = qwen3_model.generate_result(messages)
    content = qwen3_model.generate_result_stream2(messages)
    return content

def _result_query(query, sys_prompt):
    from ._utils import clean_output

    result = ""
    for chunk in _result_query_stream(query, sys_prompt):
        result += chunk
    return clean_output(result)

def enclose_string_with_quotes(content: Any) -> str:
    """Enclose a string with quotes"""
    if isinstance(content, numbers.Number):
        return str(content)
    content = str(content)
    content = content.strip().strip("'").strip('"')
    return f'"{content}"'




def list_of_list_to_csv(data: list[list]):
    return "\n".join(
        [
            ",\t".join([f"{enclose_string_with_quotes(data_dd)}" for data_dd in data_d])
            for data_d in data
        ]
    )

async def videorag_query(
    query,
    text_chunks_db,
    chunks_vdb,
    video_path_db,
    video_segments,
    video_segment_feature_vdb,
    query_param,
) -> str:
    results = await chunks_vdb.query(query)
    if not len(results):
        return PROMPTS["fail_response"]
    chunks_ids = [r["id"] for r in results]
    chunks = await text_chunks_db.get_by_ids(chunks_ids)
   
    maybe_trun_chunks = truncate_list_by_token_size(
        chunks,
        key=lambda x: x["content"],
        max_token_size = query_param.naive_max_token_for_text_unit,
    )
    logger.info(f"Truncate {len(chunks)} to {len(maybe_trun_chunks)} chunks")
    section = "-----New Chunk-----\n".join([c["content"] for c in maybe_trun_chunks])
    retreived_chunk_context = section

    query_for_visual_retrieval =  query
        

    segment_results = await video_segment_feature_vdb.query(query_for_visual_retrieval)
    
    
    visual_retrieved_segments = set()
    if len(segment_results):
        for n in segment_results:
            visual_retrieved_segments.add(n['__id__'])
    
    # caption
    
    retrieved_segments = sorted(
        visual_retrieved_segments,
        key=lambda x: (
            '_'.join(x.split('_')[:-1]), # video_name
            eval(x.split('_')[-1]) # index
        )
    )

    
    rough_captions = {}
    print("retrieved_segments :: \n", retrieved_segments)
    for s_id in retrieved_segments:
        video_name = '_'.join(s_id.split('_')[:-1])
        index = s_id.split('_')[-1]
        rough_captions[s_id] = video_segments._data[video_name][index]["content"]
   
    
    remain_segments = retrieved_segments

    keywords_for_caption = clean_output(_extract_keywords_query(query )) 
   
    caption_results = retrieved_segment_caption_kw(
        remain_segments,
        video_path_db,
        video_segments,
        keywords_for_caption,
        num_sampled_frames = query_param.retrieved_num_sampled_frames
    )

    ## data table
    text_units_section_list = [["video_name", "start_time", "end_time", "content"]]
    for s_id in caption_results:
        video_name = '_'.join(s_id.split('_')[:-1])
        index = s_id.split('_')[-1]
        start_time = eval(video_segments._data[video_name][index]["time"].split('-')[0])
        end_time = eval(video_segments._data[video_name][index]["time"].split('-')[1])
        start_time = f"{start_time // 3600}:{(start_time % 3600) // 60}:{start_time % 60}"
        end_time = f"{end_time // 3600}:{(end_time % 3600) // 60}:{end_time % 60}"
        text_units_section_list.append([video_name, start_time, end_time, caption_results[s_id]])
    text_units_context = list_of_list_to_csv(text_units_section_list)

    retreived_video_context = f"\n-----Retrieved Knowledge From Videos-----\n```csv\n{text_units_context}\n```\n"
    
    sys_prompt_temp = PROMPTS["videorag_response"]
    print("retreived_chunk_context", retreived_video_context)
    sys_prompt = sys_prompt_temp.format(
        video_data=retreived_video_context,
        chunk_data=retreived_chunk_context,      
    )
   
    # response = _result_query(query,sys_prompt )

    response = _result_query_back(query,
        sys_prompt,
    )

    return response
