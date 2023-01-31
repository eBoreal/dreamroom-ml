import math
import random

import torch
from PIL import Image, ImageOps
from diffusers import StableDiffusionInstructPix2PixPipeline,EulerAncestralDiscreteScheduler

from utils import imageStringToPil, pilToDataUrl

# Init is ran on server startup
# Load  model to GPU as a global variable under pipeline
def init():
    global model
    
    device = 0 if torch.cuda.is_available() else -1
    model_id = "timbrooks/instruct-pix2pix"
    model = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
    model.to("cuda")
    model.scheduler = EulerAncestralDiscreteScheduler.from_config(model.scheduler.config)

# from https://huggingface.co/spaces/timbrooks/instruct-pix2pix/blob/main/edit_app.py
def generate(
    input_image: Image.Image,
    instruction: str,
    steps: int,
    randomize_seed: bool,
    seed: int,
    randomize_cfg: bool,
    text_cfg_scale: float,
    image_cfg_scale: float,
    ):
    global model

    seed = random.randint(0, 100000) if randomize_seed else seed
    text_cfg_scale = round(random.uniform(6.0, 9.0), ndigits=2) if randomize_cfg else text_cfg_scale
    image_cfg_scale = round(random.uniform(1.2, 1.8), ndigits=2) if randomize_cfg else image_cfg_scale

    width, height = input_image.size
    factor = 512 / max(width, height)
    factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
    width = int((width * factor) // 64) * 64
    height = int((height * factor) // 64) * 64
    input_image = ImageOps.fit(input_image, (width, height), method=Image.Resampling.LANCZOS)

    if instruction == "":
        return {
           'seed': seed, 
            'image': input_image 
        }

    print("Launching model with image: ", type(input_image), input_image.size)
    print("prompt: ", instruction)
    print(seed)
    print(text_cfg_scale)
    print(image_cfg_scale)

    generator = torch.manual_seed(seed)
    edited_image = model(
        instruction, image=input_image,
        guidance_scale=text_cfg_scale, image_guidance_scale=image_cfg_scale,
        num_inference_steps=steps, generator=generator
    ).images[0]
    return {
        'seed': seed, 
        'text_cfg_scale': text_cfg_scale, 
        'image_cfg_scale': image_cfg_scale, 
        'image': edited_image
        }

# Inference is ran for every server call
# Reference preloaded global pipeline here. 
def inference(model_inputs:dict) -> dict:
    # Parse pipeline arguments
    instruction = model_inputs.get('prompt', None)
    steps = min(model_inputs.get('num_inference_steps', 10), 50)
    image_cfg_scale=model_inputs.get('image_guidance_scale', 2.5)
    text_cfg_scale=model_inputs.get('prompt_guidance_scale', 7)
    randomize_cfg=model_inputs.get('randomize_cfg', True)
    randomize_seed=model_inputs.get('randomize_seed', True)
    seed=model_inputs.get('seed', 42)

    # num_images_per_prompt=model_inputs.get('num_images_per_prompt', 1)

    # decode image
    image_string = model_inputs.get('imageString')

    print("Received image string", image_string[:50])

    input_image = imageStringToPil(image_string)

    print("Decoded it to", type(input_image))


    if instruction == None:
        return {'message': "No prompt provided"}

    # Run the model
    result = generate(
            input_image,
            instruction,
            steps,
            randomize_seed,
            seed,
            randomize_cfg,
            text_cfg_scale,
            image_cfg_scale
    )

    # Return the results as a dictionary
    edited_img_string = pilToDataUrl(result['image'])
    result['image'] = edited_img_string

    print("Replying to server with image: ", result['image'][:50])
    
    return result