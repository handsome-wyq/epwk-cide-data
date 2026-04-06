import requests
import json

MODEL_NAME = "qwen3:8b"  # 和你 ollama run 用的一样

prompt = "设计一个女包LOGO，只输出关键短语，用逗号分隔。"

url = "http://localhost:11434/api/generate"
payload = {
    "model": MODEL_NAME,
    "prompt": prompt,
    "stream": False,
    "options": {
        "temperature": 0.1,
        "num_predict": 64
    }
}

print("== 发起请求 ==")
print(payload)

try:
    resp = requests.post(url, json=payload, timeout=60)
    print("== HTTP 状态码 ==", resp.status_code)
    print("== 原始响应文本 ==")
    print(resp.text)
    resp.raise_for_status()
    data = resp.json()
    print("== 解析后的 JSON ==")
    print(json.dumps(data, ensure_ascii=False, indent=2))
    print("== data['response'] ==")
    print(repr(data.get("response", "")))
except Exception as e:
    print("== 出错了 ==")
    print(repr(e))
