# LLaVA service

Serve a LLaVA image capturing model as a REST service.

# Install & run

It is required to have GPU like A2 (~12Gb VRAM is required) to load llava-hf/llava-1.5-7b-hf from hugging face repository.

Command to start service
```bash
fastapi dev --port 18912 --host 0.0.0.0 src/main.py
```



