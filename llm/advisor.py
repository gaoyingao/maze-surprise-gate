import time
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:0.5b"


def ask_llm(prompt: str) -> str:
    """调用本地 Ollama 生成回复"""
    payload = {"model": MODEL_NAME, "prompt": prompt, "stream": False}
    response = requests.post(OLLAMA_URL, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()["response"].strip()


if __name__ == "__main__":
    test_prompt = (
        "你在12x12迷宫的(3,4)，怪物在(5,5)，出口在(10,10)。"
        "只回答一个方向词：上、下、左、右。不要解释。"
    )

    print("正在测试 Python 调用 Ollama...")
    start_time = time.time()
    try:
        reply = ask_llm(test_prompt)
        elapsed = time.time() - start_time

        print("=" * 40)
        print(f"模型回复内容: {reply}")
        print(f"单次调用耗时: {elapsed:.2f} 秒")
        print("=" * 40)
        print("✅ Day 7 Python 调用验证成功！")
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到 Ollama！请确保 Ollama 应用程序已在后台启动。")