import requests
import json
import os
class Qwen3TokenizerClient:
    def __init__(self, base_url :str = None):
        self.base_url = base_url or os.getenv("Tokenizer_API_BASE_URL", "http://localhost:8000/v1")
        self.encode_url = f"{self.base_url}/encode"
        self.decode_url = f"{self.base_url}/decode"
        self.batch_url = f"{self.base_url}/batch_encode"
        self.health_url = f"{self.base_url}/health"

    def check_health(self):
        """检查服务是否正常运行"""
        try:
            response = requests.get(self.health_url)
            if response.status_code == 200:
                print(f"✅ 服务状态正常: {response.json()}")
                return True
            else:
                print(f"❌ 服务检查失败: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ 无法连接到服务: {e}")
            return False

    def encode(self, text, add_special_tokens=True):
        """
        将文本转换为 Token IDs
        """
        payload = {
            "text": text,
            "add_special_tokens": add_special_tokens
        }
        
        response = requests.post(self.encode_url, json=payload)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ 编码失败: {response.text}")
            return None
    
    def batch_encode(self, texts, padding=True, max_length=None):
        """
        批量发送文本进行编码
        """
        payload = {
            "texts": texts,
            "padding": padding,
            "max_length": max_length
        }
        
        response = requests.post(self.batch_url, json=payload)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ 批量编码失败: {response.text}")
            return None

    def decode(self, token_ids, skip_special_tokens=True):
        """
        将 Token IDs 还原为文本
        """
        payload = {
            "token_ids": token_ids,
            "skip_special_tokens": skip_special_tokens
        }
        
        response = requests.post(self.decode_url, json=payload)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ 解码失败: {response.text}")
            return None

# --- 主程序入口 ---
if __name__ == "__main__":
    # 1. 初始化客户端 (如果你的 API 运行在不同端口，请修改这里)
    client = Qwen3TokenizerClient(base_url="http://127.0.0.1:8001")

    # 2. 检查服务状态
    if not client.check_health():
        print("请先启动 API 服务 (python tokenizer_api.py)")
        exit()

    print("-" * 30)

    # 3. 测试 Encode (文本 -> ID)
    input_text = "你好，Qwen3！"
    print(f"📝 原始文本: {input_text}")
    
    encode_result = client.encode(input_text)
    
    if encode_result:
        token_ids = encode_result['token_ids']
        count = encode_result['count']
        
        print(f"🔢 Token IDs: {token_ids}")
        print(f"📏 Token 长度: {count}")
        
        print("-" * 30)

        # 4. 测试 Decode (ID -> 文本)
        # 这里我们尝试还原刚才生成的 IDs
        decode_result = client.decode(token_ids, skip_special_tokens=False)
        
        if decode_result:
            restored_text = decode_result['text']
            print(f"🔄 还原结果 (含特殊标记): {restored_text}")
            
            # 尝试跳过特殊标记还原
            clean_result = client.decode(token_ids, skip_special_tokens=True)
            print(f"✨ 纯净文本: {clean_result['text']}")