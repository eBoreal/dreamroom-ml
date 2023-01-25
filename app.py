import torch
from diffusers import StableDiffusionInstructPix2PixPipeline,EulerAncestralDiscreteScheduler

from utils.img_helpers import string_to_pil, pil_to_string

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
# Reference preloaded global pipeline here. 
def inference(model_inputs:dict) -> dict:
    global pipeline
    # Parse pipeline arguments
    prompt = model_inputs.get('prompt', None)
    base64_string = model_inputs.get('image')
    image = string_to_pil(base64_string)

    if prompt == None:
        return {'message': "No prompt provided"}
    
    # Run the model
    result = pipeline(prompt, image=image, num_inference_steps=10, image_guidance_scale=1).images

    # Return the results as a dictionary
    return {'image': pil_to_string(result[0])}