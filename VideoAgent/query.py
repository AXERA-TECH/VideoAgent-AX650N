import os
import numbers
import asyncio
from typing import Any, Union
from .prompt import PROMPTS

from ._utils import logger
from ._llm import Qwen3
from ._videoutil import (
   
    retrieved_segment_caption,
)

from dotenv import load_dotenv
from transformers import AutoTokenizer


load_dotenv()




qwen3_model = Qwen3()
tiktoken_model_path = qwen3_model.download_tokenizer_files()

tiktoken_model_path = os.path.abspath(tiktoken_model_path)



try:
    tiktoken_model_p = AutoTokenizer.from_pretrained(
        tiktoken_model_path,
        trust_remote_code=True,
        local_files_only=True  # <--- 关键参数
    )
except Exception as e:
    print(f"加载本地 tokenizer 失败: {e}")
    print(f"请检查路径是否存在: {tiktoken_model_path}")
    raise e




def truncate_list_by_token_size(list_data: list, key: callable, max_token_size: int):
    """Truncate a list of data by token size"""
    if max_token_size <= 0:
        return []
    tokens = 0
    for i, data in enumerate(list_data):
        tokens += len(tiktoken_model_p.encode(key(data)))
        if tokens > max_token_size:
            return list_data[:i]
    return list_data



def _result_query(query,sys_prompt):

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": query}
    ]
     
    content = qwen3_model.generate_result(messages)
    return content

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
    results = await chunks_vdb.query(query, top_k = query_param.top_k)
    if not len(results):
        return PROMPTS["fail_response"]
    chunks_ids = [r["id"] for r in results]
    chunks = await text_chunks_db.get_by_ids(chunks_ids)
    # print("chunks :\n", chunks)
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
    
    print(f"Retrieved Segments {segment_results}")
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

    caption_results = retrieved_segment_caption(
        remain_segments,
        video_path_db,
        video_segments,
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
    # print(retreived_video_context)
    sys_prompt_temp = PROMPTS["videorag_response"]
        
    sys_prompt = sys_prompt_temp.format(
        video_data=retreived_video_context,
        chunk_data=retreived_chunk_context,
        
    )
   
    response = _result_query(query,
        sys_prompt,
    )
    return response
