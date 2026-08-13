import os
import sys
import json
import shutil
import asyncio
import multiprocessing
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
from typing import Callable, Dict, List, Optional, Type, Union, cast
from transformers import AutoModel, AutoTokenizer
import torch
import queue
from ._storage import (
    JsonKVStorage,
    NanoVectorDBStorage,
    NanoVectorDBVideoSegmentStorage,
    # NetworkXStorage,
)
from ._utils import (
    always_get_an_event_loop,
    logger,
)
from .base import (
    # BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    StorageNameSpace,
    QueryParam,
)
from ._videoutil import(
    split_video,
    speech_to_text,
    segment_caption,
    merge_segment_information,
    saving_video_segments,
    preprocess_video,
    retrieved_segment_caption_kw,
)

from .chunk import (
    get_chunks,
    chunking_by_video_segments,
)

from .query import (
    videorag_query,
    _result_query_stream,
    _extract_keywords_query,
    truncate_list_by_token_size,
    list_of_list_to_csv,
)
from .prompt import PROMPTS




@dataclass
class VideoRAG:
    working_dir: str = field(
        default_factory=lambda: f"./videorag_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )
    
    # video
    threads_for_split: int = 10
    video_segment_length: int = 10 # seconds
    rough_num_frames_per_segment: int = 5 # frames

    video_output_format: str = "mp4"
    audio_output_format: str = "wav"
    # preprocessing: resize and resample input video before indexing
    preprocess_target_width: int = 384
    preprocess_target_height: int = 384
    preprocess_target_fps: int = 5

    video_embedding_batch_num: int = 2
    segment_retrieval_top_k: int = 1
    video_embedding_dim: int = 2048  # qwen3-vl-embedding 输出维度
    embedding_batch_num : int = 2
    # query
    retrieval_topk_chunks: int = 1
    query_better_than_threshold: float = 0.2
    
    enable_local: bool = True
    enable_naive_rag: bool = True
    chunk_token_size: int = 1000

    entity_extract_max_gleaning: int = 1
    entity_summary_to_max_tokens: int = 500

    key_string_value_json_storage_cls: Type[BaseKVStorage] = JsonKVStorage
    vector_db_storage_cls: Type[BaseVectorStorage] = NanoVectorDBStorage
    vs_vector_db_storage_cls: Type[BaseVectorStorage] = NanoVectorDBVideoSegmentStorage
    vector_db_storage_cls_kwargs: dict = field(default_factory=dict)
    enable_llm_cache: bool = True

    always_create_working_dir: bool = True

    def __post_init__(self):
        if not os.path.exists(self.working_dir) and self.always_create_working_dir:
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)

        self.video_path_db = self.key_string_value_json_storage_cls(
            namespace="video_path", global_config=asdict(self)
        )
        
        self.video_segments = self.key_string_value_json_storage_cls(
            namespace="video_segments", global_config=asdict(self)
        )

        self.text_chunks = self.key_string_value_json_storage_cls(
            namespace="text_chunks", global_config=asdict(self)
        )

        self.chunks_vdb = self.vector_db_storage_cls(
            namespace="chunks_vdb", global_config=asdict(self)
        )

        self.video_segment_feature_vdb = (
            self.vs_vector_db_storage_cls(
                namespace="video_segment_feature",
                global_config=asdict(self),
            )
        )


    def insert_video(self, video_path_list=None):
        loop = always_get_an_event_loop()
        total = len(video_path_list)
        for i, video_path in enumerate(video_path_list, start=1):
            # Step0: check the existence
            video_name = os.path.basename(video_path).split('.')[0]
            logger.info(f"[{i}/{total}] 🎬 Processing: {video_name}")
            if video_name in self.video_segments._data:
                logger.info(f"Find the video named {os.path.basename(video_path)} in storage and skip it.")
                continue

            logger.info(f"[{i}/{total}]   ├ Step 1/7: Preprocessing video (resize/resample)...")
            video_output_path = preprocess_video(
                video_path,
                self.preprocess_target_width,
                self.preprocess_target_height,
                self.preprocess_target_fps,
                self.video_output_format,
            )

            logger.info(f"[{i}/{total}]   ├ Step 2/7: Saving video path metadata...")
            loop.run_until_complete(self.video_path_db.upsert(
                {video_name: video_output_path}
            ))
            loop.run_until_complete(self.video_path_db.index_done_callback())

            logger.info(f"[{i}/{total}]   ├ Step 3/7: Splitting video into segments + extracting audio...")
            segment_index2name, segment_times_info = split_video(
                video_output_path,
                self.working_dir,
                self.video_segment_length,
                self.rough_num_frames_per_segment,
                self.audio_output_format,
            )

            logger.info(f"[{i}/{total}]   ├ Step 4/7: Speech recognition (ASR)...")
            transcripts = speech_to_text(
                video_name,
                self.working_dir,
                segment_index2name,
                self.audio_output_format
            )

            captions = dict()
            error_queue = queue.Queue()

            logger.info(f"[{i}/{total}]   ├ Step 5/7: Saving video segments to cache...")
            saving_video_segments(video_name,
                    video_output_path,
                    self.working_dir,
                    segment_index2name,
                    segment_times_info,
                    error_queue,
                    self.video_output_format,)

            logger.info(f"[{i}/{total}]   ├ Step 6/7: Generating captions with VLM...")
            segment_caption(
                video_name,
                video_output_path,
                segment_index2name,
                transcripts,
                segment_times_info,
                captions,
                error_queue,
            )

            while not error_queue.empty():
                error_message = error_queue.get()
                with open('error_log_videorag.txt', 'a', encoding='utf-8') as log_file:
                    log_file.write(f"Video Name:{video_name} Error processing:\n{error_message}\n\n")
                raise RuntimeError(error_message)

            logger.info(f"[{i}/{total}]   ├ Step 7/7: Merging segment info & encoding features...")
            segments_information = merge_segment_information(
                            segment_index2name,
                            segment_times_info,
                            transcripts,
                            captions,
                        )

            loop.run_until_complete(self.video_segments.upsert(
                {video_name: segments_information}
            ))
            loop.run_until_complete(self.video_segments.index_done_callback())

            loop.run_until_complete(self.video_segment_feature_vdb.upsert(
                video_name,
                segment_index2name,
                self.video_output_format,
            ))
            loop.run_until_complete(self.video_segment_feature_vdb.index_done_callback())

            video_segment_cache_path = os.path.join(self.working_dir, '_cache', video_name)
            if os.path.exists(video_segment_cache_path):
                shutil.rmtree(video_segment_cache_path)

            # 添加垃圾回收和内存清理
            import gc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            logger.info(f"[{i}/{total}] ✅ Completed: {video_name}")

        logger.info("📝 Chunking and building text index for all videos...")
        loop.run_until_complete(self.ainsert(self.video_segments._data, asdict(self)))
        logger.info("🎉 All indexing completed.")
    async def ainsert(self, new_video_segment, global_configs):
        logger.info("  → Tokenizing and chunking video segments...")
        inserting_chunks = get_chunks(
            new_videos=new_video_segment,
            chunk_func=chunking_by_video_segments,
            max_token_size=self.chunk_token_size,
        )

        logger.info("  → Filtering already-indexed chunks...")
        _add_chunk_keys = await self.text_chunks.filter_keys(
            list(inserting_chunks.keys())
        )
        inserting_chunks = {
            k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys
        }
        if not len(inserting_chunks):
            logger.warning(f"All chunks are already in the storage")
            return
        logger.info(f"  → Inserting {len(inserting_chunks)} new chunks into vector DB...")

        logger.info("  → Encoding and upserting chunks to vector DB...")
        await self.chunks_vdb.upsert(inserting_chunks)
        await self.chunks_vdb.index_done_callback()

        logger.info("  → Saving text chunks to KV store...")
        await self.text_chunks.upsert(inserting_chunks)
        await self.text_chunks.index_done_callback()
    def query(self, query: str, param: QueryParam = QueryParam()):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery(query, param))  

    async def aquery(self, query: str, param: QueryParam = QueryParam()):

        response = await videorag_query(
            query,
            self.text_chunks,
            self.chunks_vdb,
            self.video_path_db,
            self.video_segments,
            self.video_segment_feature_vdb,
            param
        )
        return response

    def query_stream(self, query: str, param: QueryParam = QueryParam()):
        sys_prompt = self._prepare_query_context(query, param)
        for chunk in _result_query_stream(query, sys_prompt):
            yield chunk

    def _prepare_query_context(self, query: str, param: QueryParam) -> str:
        loop = always_get_an_event_loop()

        results = loop.run_until_complete(self.chunks_vdb.query(query))
        if not len(results):
            return PROMPTS["fail_response"]

        chunks_ids = [r["id"] for r in results]
        chunks = loop.run_until_complete(self.text_chunks.get_by_ids(chunks_ids))

        maybe_trun_chunks = truncate_list_by_token_size(
            chunks,
            key=lambda x: x["content"],
            max_token_size=param.naive_max_token_for_text_unit,
        )
        section = "-----New Chunk-----\n".join([c["content"] for c in maybe_trun_chunks])
        retreived_chunk_context = section

        segment_results = loop.run_until_complete(
            self.video_segment_feature_vdb.query(query)
        )

        visual_retrieved_segments = set()
        if len(segment_results):
            for n in segment_results:
                visual_retrieved_segments.add(n['__id__'])

        retrieved_segments = sorted(
            visual_retrieved_segments,
            key=lambda x: (
                '_'.join(x.split('_')[:-1]),
                eval(x.split('_')[-1])
            )
        )

        remain_segments = retrieved_segments
        keywords_for_caption = _extract_keywords_query(query)

        caption_results = retrieved_segment_caption_kw(
            remain_segments,
            self.video_path_db,
            self.video_segments,
            keywords_for_caption,
            num_sampled_frames=param.retrieved_num_sampled_frames
        )

        text_units_section_list = [["video_name", "start_time", "end_time", "content"]]
        for s_id in caption_results:
            video_name = '_'.join(s_id.split('_')[:-1])
            index = s_id.split('_')[-1]
            start_time = eval(self.video_segments._data[video_name][index]["time"].split('-')[0])
            end_time = eval(self.video_segments._data[video_name][index]["time"].split('-')[1])
            start_time = f"{start_time // 3600}:{(start_time % 3600) // 60}:{start_time % 60}"
            end_time = f"{end_time // 3600}:{(end_time % 3600) // 60}:{end_time % 60}"
            text_units_section_list.append([video_name, start_time, end_time, caption_results[s_id]])
        text_units_context = list_of_list_to_csv(text_units_section_list)

        retreived_video_context = f"\n-----Retrieved Knowledge From Videos-----\n```csv\n{text_units_context}\n```\n"

        sys_prompt_temp = PROMPTS["videorag_response"]
        return sys_prompt_temp.format(
            video_data=retreived_video_context,
            chunk_data=retreived_chunk_context,
        )

