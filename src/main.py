import fastapi
import PIL
import torch
import transformers
import io


app = fastapi.FastAPI()

model_id = "llava-hf/llava-1.5-7b-hf"
quantization_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)
pipe = transformers.pipeline("image-to-text", model=model_id,
                             model_kwargs={"quantization_config": quantization_config})


@app.get("/")
def read_item(prompt: str, img: fastapi.UploadFile, max_new_tokens: int = 2000) -> str:
    image = PIL.Image.open(io.BytesIO(img.file.read()))
    if image is None:
        raise RuntimeError(f"Error, an image doesnt exists")

    outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": max_new_tokens})
    return outputs[0]["generated_text"]
