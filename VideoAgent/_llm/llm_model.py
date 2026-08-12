import os
from openai import OpenAI
from modelscope.hub.snapshot_download import snapshot_download
from modelscope.hub.file_download import download_file
from modelscope.hub.file_download import model_file_download


class Qwen3:
    def __init__(self, model_name: str = None, base_url: str = None, api_key: str = None):
        self.model_name = model_name or os.getenv("LLM_MODEL_NAME", "Qwen3-4B-Instruct-2507")
        self.base_url = base_url or os.getenv("LLM_API_BASE_URL", "http://localhost:8000/v1")
        self.api_key = api_key or os.getenv("LLM_API_KEY", "EMPTY")
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

        print(f"LLM 模型: {self.model_name}")

    def generate_result(self, messages, max_new_tokens: int = 1024) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_new_tokens,
            extra_body={"enable_thinking": False},
        )
        return response.choices[0].message.content

    def count_tokens(self, messages):
        """
        通过调用 API 来计算 Token 数量，而不需要生成长文本。
        
        原理：发送请求但限制最大生成长度为 0 (或 1)，
        服务器仍会处理 Prompt 并返回 usage.prompt_tokens。
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=1,  # 关键：限制生成极少 token，节省时间且只为了获取统计
                temperature=0, # 关闭随机性
                stream=False   # 必须是非流式才能获取完整 usage
            )
            
            # 从返回结果中提取 Prompt (输入) 的 Token 数
            prompt_tokens = response.usage.prompt_tokens
            
            # 如果需要总消耗预估 (输入 + 最小输出)
            total_tokens = response.usage.total_tokens
            
            return prompt_tokens

        except Exception as e:
            print(f"API Token 计算失败: {e}")
            
            return None
        
    def download_tokenizer_files(self):
        """
        仅下载指定的三个 tokenizer 文件:
        tokenizer_config.json, tokenizer.json, vocab.json
        """
        
        folder_name = "tokenizer_model"
        
        # 1. 稳健地获取当前脚本所在目录 (兼容脚本运行和交互式环境)
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            # 如果在 Jupyter 或交互式 shell 中 __file__ 未定义
            current_dir = os.getcwd()
            
        target_dir = os.path.join(current_dir, folder_name)

        # 2. 创建目录
        if not os.path.exists(target_dir):
            try:
                os.makedirs(target_dir)
                # print(f"成功创建目录: {target_dir}")
            except OSError as e:
                print(f"无法创建目录 {target_dir}: {e}")
                return None
        else:
            print(f"目录已存在: {target_dir}")

        # 3. 定义必须下载的文件列表
        # 注意：这三个文件是强依赖，缺一不可（对于使用此配置的模型）
        required_files = [
            'tokenizer_config.json',
            'tokenizer.json',
            'vocab.json'
        ]
        
        # print(f"开始下载模型 [{self.model_name}] 的指定分词器文件...")
        # print(f"目标路径: {target_dir}")
        
        success_count = 0
        final_storage_dir = os.path.join(target_dir, self.model_name)
        for file_name in required_files:
            local_file = os.path.join(final_storage_dir, file_name)
            if os.path.exists(local_file):
                # print(f"已存在，跳过下载: {file_name}")
                if final_storage_dir is None:
                    final_storage_dir = target_dir
                success_count += 1
                continue
            try:
                # print(f"正在下载: {file_name} ...")
                
                # 调用 ModelScope 下载接口
                # local_dir_use_symlinks=False 确保文件实体被复制到 target_dir，而不是软链接
                model_file_download(
                    model_id=self.model_name,
                    file_path=file_name,
                    cache_dir=target_dir,
                    # local_dir_use_symlinks=False
                )
               

            except Exception as e:
                error_str = str(e)
                if "404" in error_str or "not found" in error_str.lower():
                    print(f" 跳过: {file_name} (模型仓库中不存在)")
                    continue
                else:
                    print(f"下载失败: {file_name} - {error_str}")
                    if file_name == 'tokenizer_config.json':
                        return None

        print(f"成功下载 {final_storage_dir} 个文件")
        # if final_storage_dir:
            # Check if the constructed path actually exists
            # ModelScope may normalize model names (e.g., dots to underscores)
        if not os.path.exists(final_storage_dir):
            parent_dir = os.path.dirname(final_storage_dir)
            dir_name = os.path.basename(final_storage_dir)

            if os.path.exists(parent_dir):
                # Look for a directory that contains the tokenizer files
                # by checking which subdirectory has tokenizer_config.json
                found = False
                required_file = 'tokenizer_config.json'

                for subdir in os.listdir(parent_dir):
                    subdir_path = os.path.join(parent_dir, subdir)
                    if os.path.isdir(subdir_path):
                        # Check if this directory contains the required tokenizer file
                        if os.path.exists(os.path.join(subdir_path, required_file)):
                            final_storage_dir = subdir_path
                            print(f"Found tokenizer in: {final_storage_dir}")
                            found = True
                            return final_storage_dir
                            # break

                if not found:
                    # Fallback: list available directories for debugging
                    available = [d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]
                    print(f"Available directories in {parent_dir}: {available}")
                    print(f"ERROR: Could not find tokenizer files in any subdirectory")
                    
            
            
            
            
            
            # if not os.path.exists(final_storage_dir):
            #     # Try to find the directory with normalized name (dots replaced by underscores)
            #     parent_dir = os.path.dirname(final_storage_dir)
            #     expected_dir_name = os.path.basename(final_storage_dir)
            #     normalized_dir_name = expected_dir_name.replace(".", "_")

            #     if os.path.exists(parent_dir):
            #         normalized_path = os.path.join(parent_dir, normalized_dir_name)
            #         if os.path.exists(normalized_path):
            #             final_storage_dir = normalized_path
            #             print(f"Found normalized path: {final_storage_dir}")

            # print(f"下载完成。存储路径: {final_storage_dir}")
            # return final_storage_dir
        else:
            # print("\n未成功下载任何有效文件。")
            return final_storage_dir

