  "Providers": [
    {
      "name": "runpod",
      "api_base_url": "https://myhost/v1/chat/completions",
      "api_key": "sk-sksksksksk",
      "models": ["Qwen/Qwen3-Coder-30B-A3B-Instruct"]
    }
  ],
  "Router": {
    "default": "runpod,Qwen/Qwen3-Coder-30B-A3B-Instruct",
    "background": "",
    "think": "",
    "longContext": "",
    "longContextThreshold": 60000,
    "webSearch": ""
  }