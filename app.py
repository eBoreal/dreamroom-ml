import math
import random

from PIL import Image, ImageOps

import torch
from diffusers import StableDiffusionInstructPix2PixPipeline,EulerAncestralDiscreteScheduler

from utils import dataUrlToPil, pilToDataUrl

# # Init is ran on server startup
# # Load  model to GPU as a global variable under pipeline
def init():
    global model
    
    device = 0 if torch.cuda.is_available() else -1
    model_id = "timbrooks/instruct-pix2pix"
    model = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
    model.to("cuda")
    model.scheduler = EulerAncestralDiscreteScheduler.from_config(model.scheduler.config)

def generate(
    prompt: str,
    input_image: Image.Image,
    steps: int,
    randomize_seed: bool,
    seed: int,
    randomize_cfg: bool,
    text_cfg_scale: float,
    image_cfg_scale: float
    ):
    global model

    seed = random.randint(0, 100000) if randomize_seed else seed
    text_cfg_scale = round(random.uniform(6.0, 9.0), ndigits=2) if randomize_cfg else text_cfg_scale
    image_cfg_scale = round(random.uniform(1.2, 1.8), ndigits=2) if randomize_cfg else image_cfg_scale

    # width, height = input_image.size
    # factor = 512 / max(width, height)
    # factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
    # width = int((width * factor) // 64) * 64
    # height = int((height * factor) // 64) * 64
    # input_image = ImageOps.fit(input_image, (width, height), method=Image.Resampling.LANCZOS)

    generator = torch.manual_seed(seed)

    edited_image = model(prompt, image=input_image, 
            num_inference_steps=steps, 
            image_guidance_scale=image_cfg_scale, 
            guidance_scale=text_cfg_scale, generator=generator).images[0]

    return {seed:seed, text_cfg_scale:text_cfg_scale, 
        image_cfg_scale:image_cfg_scale, "image_Url": pilToDataUrl(edited_image)}


# Inference is ran for every server call
# Reference preloaded global pipeline here. 
def inference(model_inputs:dict) -> dict:
    # Parse pipeline arguments
    prompt = model_inputs.get('prompt', None)
    steps = model_inputs.get('num_inference_steps', 20)
    image_cfg_scale=model_inputs.get('image_guidance_scale', 2)
    text_cfg_scale=model_inputs.get('prompt_guidance_scale', 7)
    just_test_img=model_inputs.get('test_mode', False)
    seed=model_inputs.get('seed', None)
    randomize_cfg=model_inputs.get('randomize_cfg', False)
    randomize_seed=model_inputs.get('randomize_seed', False)
    # num_images_per_prompt=model_inputs.get('num_images_per_prompt', 1)

    # decode image
    base64_string = model_inputs.get('image')
    image = dataUrlToPil(base64_string)

    print("Input image of type: ", type(image))

    if prompt == None:
        return {'message': "No prompt provided"}

    if just_test_img:
        # just test sending & getting back images
        results = {seed: seed, text_cfg_scale:text_cfg_scale, 
            image_cfg_scale:image_cfg_scale, "imageUrl": pilToDataUrl(image)}

    else:
        # Run the model
        results = generate(
                prompt,
                image,
                steps,
                randomize_seed,
                seed,
                randomize_cfg,
                text_cfg_scale,
                image_cfg_scale
        )

    # Return the results as a dictionary
    
    return results