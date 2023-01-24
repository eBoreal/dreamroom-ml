import PIL
import requests
import torch
from PIL import Image
from io import BytesIO
import base64

from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler


# Init is ran on server startup
# Load  model to GPU as a global variable under pipeline
def init():
    global pipeline
    
    device = 0 if torch.cuda.is_available() else -1
    model_id = "timbrooks/instruct-pix2pix"
    pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
    pipeline.to("cuda")
    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)


# Inference is ran for every server call
# Reference  preloaded global pipeline here. 
def inference(model_inputs:dict) -> dict:
    global pipeline
    # Parse pipeline arguments
    prompt = model_inputs.get('prompt', None)
    image_base_64 = model_inputs.get('image', None)
    image = Image.open(BytesIO(base64.b64decode(image_base_64))).convert("RGB")

    if prompt == None:
        return {'message': "No prompt provided"}
    
    # Run the model
    result = pipeline(prompt, image=image, num_inference_steps=10, image_guidance_scale=1).images

    # Return the results as a dictionary
    return result[0]
