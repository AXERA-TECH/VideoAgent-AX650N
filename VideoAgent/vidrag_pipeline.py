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
)

from .chunk import (
    get_chunks,
    chunking_by_video_segments,
)

from .query import (
    videorag_query
)




@dataclass
class VideoRAG:
    working_dir: str = field(
        default_factory=lambda: f"./videorag_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )
    
    # video
    threads_for_split: int = 10
    video_segment_length: int = 20 # seconds
    rough_num_frames_per_segment: int = 10 # frames

    video_output_format: str = "mp4"
    audio_output_format: str = "mp3"
    # preprocessing: resize and resample input video before indexing
    preprocess_target_width: int = 384
    preprocess_target_height: int = 384
    preprocess_target_fps: int = 5
    video_embedding_batch_num: int = 2
    segment_retrieval_top_k: int = 5
    video_embedding_dim: int = 2048  # qwen3-vl-embedding 输出维度
    embedding_batch_num : int = 2
    # query
    retrieval_topk_chunks: int = 2
    query_better_than_threshold: float = 0.2
    
    # graph mode
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
        for video_path in video_path_list:
            # Step0: check the existence
            video_name = os.path.basename(video_path).split('.')[0]
            if video_name in self.video_segments._data:
                logger.info(f"Find the video named {os.path.basename(video_path)} in storage and skip it.")
                continue
           
            
            video_output_path = preprocess_video(
                video_path,
                self.preprocess_target_width,
                self.preprocess_target_height,
                self.preprocess_target_fps,
                self.video_output_format,
            )

            loop.run_until_complete(self.video_path_db.upsert(
                {video_name: video_output_path}
            ))
            loop.run_until_complete(self.video_path_db.index_done_callback())

            segment_index2name, segment_times_info = split_video(
                video_output_path,
                self.working_dir,
                self.video_segment_length,
                self.rough_num_frames_per_segment,
                self.audio_output_format,
            )

            transcripts = speech_to_text(
                video_name,
                self.working_dir,
                segment_index2name,
                self.audio_output_format
            )

            captions = dict()
            error_queue = queue.Queue()

          

            saving_video_segments(video_name,
                    video_output_path,
                    self.working_dir,
                    segment_index2name,
                    segment_times_info,
                    error_queue,
                    self.video_output_format,)

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
        
        loop.run_until_complete(self.ainsert(self.video_segments._data, asdict(self)))
            

    async def ainsert(self, new_video_segment, global_configs):
        inserting_chunks = get_chunks(
            new_videos=new_video_segment,
            chunk_func=chunking_by_video_segments,
            max_token_size=self.chunk_token_size,
        )
        
        
        _add_chunk_keys = await self.text_chunks.filter_keys(
            list(inserting_chunks.keys())
        )
        inserting_chunks = {
            k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys
        }
        if not len(inserting_chunks):
            logger.warning(f"All chunks are already in the storage")
            return
        logger.info(f"[New Chunks] inserting {len(inserting_chunks)} chunks")
        
        logger.info("Insert chunks for naive RAG")
        await self.chunks_vdb.upsert(inserting_chunks)
        await self.chunks_vdb.index_done_callback()

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

