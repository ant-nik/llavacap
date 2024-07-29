import requests
import PIL
import torch
import transformers
import tempfile
import os
import pathlib
import shutil


model_id = "llava-hf/llava-1.5-7b-hf"
images_url = "https://drive.usercontent.google.com/u/0/uc?id=1IG6CXJipcApR34xtKWJpxm03Ud-5pwIx&export=download"
zipname = "images.zip"
images_dir = "images"
result_dir = "result"
output_file = "result"

quantization_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)
pipe = transformers.pipeline("image-to-text", model=model_id,
                             model_kwargs={"quantization_config": quantization_config})

max_new_tokens = 200
prompt = "USER: <image>\nFind construction entities on the image. "
"Split answer in two sections LIST and EXPLANATION. Put detected object "
"to LIST section. Put explanation of the answer into EXPLANATION section.\nASSISTANT:"

with tempfile.TemporaryDirectory() as name:
    filename = os.path.join(name, zipname)
    with open(filename, 'wb') as file:
        file.write(requests.get(images_url, stream=False).content)
    path_to_images = os.path.join(name, images_dir)
    shutil.unpack_archive(filename, path_to_images)

    result_path = os.path.join(name, result_dir)
    pathlib.Path(result_path).mkdir(parents=True, exist_ok=True)
    images = next(os.walk(path_to_images), (None, None, []))[2]
    for image_name in images:
        image = PIL.Image.open(os.path.join(path_to_images, image_name))
        if image is None:
            raise RuntimeError(f"Error, an image with a path {image_name} doesnt exists")

        outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 2000})
        with open(os.path.join(result_path, os.path.basename(image_name) + ".txt"), "w") as caption:
            caption.write(outputs[0]["generated_text"])

    shutil.make_archive(base_name=output_file, format="zip", root_dir=result_path)

