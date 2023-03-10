import math
import random

from PIL import Image, ImageOps

import torch
from diffusers import StableDiffusionInstructPix2PixPipeline,EulerAncestralDiscreteScheduler

from utils import stringToPil, pilToString

def init(
):
    """Load  model to GPU as a global variable under pipeline
    """
    global model
    
    model_id = "timbrooks/instruct-pix2pix"
    model = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
    
    if torch.cuda.is_available():
        model.to("cuda")
        
    model.scheduler = EulerAncestralDiscreteScheduler.from_config(model.scheduler.config)

def preprocess(
    image: str
)->Image.Image:
    """ Resize the image if needed. 
        Args:
            image_64_string (str) 
        Returns: 
            Image.Image
    """
    width, height = image.size
    factor = 512 / max(width, height)
    factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
    width = int((width * factor) // 64) * 64
    height = int((height * factor) // 64) * 64
    input_image = ImageOps.fit(image, (width, height), method=Image.Resampling.LANCZOS)

    return input_image



def generate(
    prompt: str,
    input_image: Image.Image,
    steps: int,
    randomize_seed: bool,
    seed: int,
    randomize_cfg: bool,
    text_cfg_scale: float,
    image_cfg_scale: float,
    num_images_per_prompt: int
)->list:
    """Runs the model to generate the edited images. 

    Args:
        prompt (str)
        input_image (Image.Image)
        steps (int)
        randomize_seed (bool)
        seed (int)
        randomize_cfg (bool)
        text_cfg_scale (float)
        image_cfg_scale (float)
        num_images_per_prompt (int)

    Returns:
        dict: generated images objects
    """
    global model

    seed = random.randint(0, 100000) if randomize_seed else seed
    text_cfg_scale = round(random.uniform(6.0, 9.0), ndigits=2) if randomize_cfg else text_cfg_scale
    image_cfg_scale = round(random.uniform(1.2, 1.8), ndigits=2) if randomize_cfg else image_cfg_scale

    generator = torch.manual_seed(seed)

    # debugging
    print("Running model with params: ", 
        {'prompt': prompt, 
        'steps': steps, 
        'image_cfg_scale': image_cfg_scale, 
        'text_cfg_scale': text_cfg_scale,
        'image_size': input_image.size,
        "num_images_per_prompt": num_images_per_prompt})

    res = []
    i = 0
    #for (img_cfg, text_cfg) in grid_search:
    i+=1

    img = model(prompt, image=input_image, 
                num_inference_steps=steps, 
                image_guidance_scale=image_cfg_scale, 
                guidance_scale=text_cfg_scale, 
                generator=generator).images[0]
    
    return {
        'seed':seed, 
        'text_cfg_scale':text_cfg_scale, 
        'image_cfg_scale':image_cfg_scale, 
        "steps": steps,
        'image': pilToString(img)
        }


# Inference is ran for every server call
# Reference preloaded global pipeline here. 
def inference(
    model_inputs:dict
)->list:
    """ Runs inference pipeline.

    Args: 
        model_inputs (dict): a key value store of the inputs for pix2pix
    Returns:
        list of generated images objects
    """
    
    # Parse pipeline arguments
    # Official model inputs
    prompt = model_inputs.get('prompt', None)
    steps = model_inputs.get('steps', 20)
    image_cfg_scale=model_inputs.get('image_cfg_scale', 1.5)
    text_cfg_scale=model_inputs.get('text_cfg_scale', 7)
    seed=model_inputs.get('seed', 42)
    randomize_cfg=model_inputs.get('randomize_cfg', False)
    randomize_seed=model_inputs.get('randomize_seed', True)
    num_images_per_prompt=model_inputs.get('num_images', 1)
        
    # Custom
    just_test_img=model_inputs.get('test_mode', False)
    toDataUrl=model_inputs.get('toDataUrl', False)

    # decode image
    base64_string = model_inputs.get('image')
    image = stringToPil(base64_string)

    if prompt == None:
        return {'message': "No prompt provided"}

    results = []
    if just_test_img:
        # just test sending & getting back images
        results.append({
            "seed": seed, 
            "text_cfg_scale":text_cfg_scale, 
            "image_cfg_scale":image_cfg_scale, 
            "steps": steps,
            'image': pilToString(image, dataUrl=toDataUrl)
            })

    else:
        # preprocessing step
        preprocessed_img = preprocess(image)

        # generate different samples
        sample_variants = [
            (image_cfg_scale, text_cfg_scale),
            (round(image_cfg_scale-.15, 3), text_cfg_scale),
            (round(image_cfg_scale-.3, 3), text_cfg_scale),
            (round(image_cfg_scale+.15, 3), text_cfg_scale)

        ]
        
        for i, (img_cfg, text_cfg) in enumerate(sample_variants):
            print("Running for sample: ", i)
            i+=1
            # Run the model
            results.append(
                generate(
                    prompt=prompt,
                    input_image=preprocessed_img,
                    steps=steps,
                    randomize_seed=randomize_seed,
                    seed=seed,
                    randomize_cfg=randomize_cfg,
                    text_cfg_scale=text_cfg,
                    image_cfg_scale=img_cfg,
                    num_images_per_prompt=num_images_per_prompt
                    )
                )
    return results