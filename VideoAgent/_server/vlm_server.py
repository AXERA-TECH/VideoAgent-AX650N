import subprocess
import sys
import os
from dotenv import load_dotenv

load_dotenv()
MODEL_PATH = os.getenv("VLM_MODEL_PATH", "")
MODEL_NAME = os.getenv("VLM_MODEL_NAME", "")
GPU_DEVICE_ID = os.getenv("CUDA_VISIBLE_DEVICES", "1")  # Default to GPU 0 if not specified
PORT = os.getenv("VLM_API_PORT", 8008)
def start_llm_server():
    """
    Start the vLLM server with Qwen3 model on specified GPU
    """
    # Set CUDA_VISIBLE_DEVICES to specify which GPU to use
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = GPU_DEVICE_ID
    
    cmd = [
        "vllm", "serve", MODEL_PATH,
        "--trust-remote-code",
        "--dtype", "half",
        "--port", str(PORT),
        "--max-model-len", "4096",
        "--served-model-name", MODEL_NAME,
        "--gpu-memory-utilization", "0.35",
        # "--max-num-batched-tokens", "1024"
    ]
    
    print(f"Starting vLLM server on GPU {GPU_DEVICE_ID}")
    # print("Command:", " ".join(cmd))
    
    try:
        # Execute the command with the specified GPU environment
        process = subprocess.Popen(cmd, env=env)
        
        # Wait for the process to complete (this will run indefinitely since it's a server)
        process.wait()
    except KeyboardInterrupt:
        print("\nServer stopped by user.")
        process.terminate()
        process.wait()

if __name__ == "__main__":
    start_llm_server()