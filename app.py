import math
import random

from PIL import Image, ImageOps

import torch
from diffusers import StableDiffusionInstructPix2PixPipeline,EulerAncestralDiscreteScheduler

from utils import stringToPil, pilToString

# # Init is ran on server startup
# Load  model to GPU as a global variable under pipeline
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

    width, height = input_image.size
    factor = 512 / max(width, height)
    factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
    width = int((width * factor) // 64) * 64
    height = int((height * factor) // 64) * 64
    input_image = ImageOps.fit(input_image, (width, height), method=Image.Resampling.LANCZOS)

    generator = torch.manual_seed(seed)

    print("Running model with params: ", {'prompt': prompt, 'steps': steps, 'image_cfg_scale': image_cfg_scale, 'text_cfg_scale': text_cfg_scale})
    print("Image: ", input_image.size)

    edited_image = model(prompt, image=input_image, 
            num_inference_steps=steps, 
            image_guidance_scale=image_cfg_scale, 
            guidance_scale=text_cfg_scale, generator=generator).images[0]

    return {
        seed:seed, 
        text_cfg_scale:text_cfg_scale, 
        image_cfg_scale:image_cfg_scale, 
        'image': edited_image
        }


# Inference is ran for every server call
# Reference preloaded global pipeline here. 
def inference(
    model_inputs:dict
    ) -> dict:
    # Parse pipeline arguments
    # Official model inputs
    prompt = model_inputs.get('prompt', None)
    steps = model_inputs.get('steps', 30)
    image_cfg_scale=model_inputs.get('image_cfg_scale', 1.5)
    text_cfg_scale=model_inputs.get('text_cfg_scale', 7)
    seed=model_inputs.get('seed', None)
    randomize_cfg=model_inputs.get('randomize_cfg', False)
    randomize_seed=model_inputs.get('randomize_seed', False)
        
    # Custom
    just_test_img=model_inputs.get('test_mode', False)
    toDataUrl=model_inputs.get('toDataUrl', False)

    # decode image
    base64_string = model_inputs.get('image')
    image = stringToPil(base64_string)

    print("Input image of type: ", type(image))

    if prompt == None:
        return {'message': "No prompt provided"}

    if just_test_img:
        # just test sending & getting back images
        results = {
            seed: seed, 
            text_cfg_scale:text_cfg_scale, 
            image_cfg_scale:image_cfg_scale, 
            'image': pilToString(image, dataUrl=toDataUrl)
            }

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
                image_cfg_scale,
        )

        edited_image = results.image
        results.image=pilToString(edited_image, dataUrl=toDataUrl)

    # Return the results as a dictionary
    
    return results