# import torch
# from diffusers import StableDiffusionInstructPix2PixPipeline,EulerAncestralDiscreteScheduler

from utils import dataUrlToPil, pilToDataUrl

# # Init is ran on server startup
# # Load  model to GPU as a global variable under pipeline
# def init():
#     global model
    
#     device = 0 if torch.cuda.is_available() else -1
#     model_id = "timbrooks/instruct-pix2pix"
#     model = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
#     model.to("cuda")
#     model.scheduler = EulerAncestralDiscreteScheduler.from_config(model.scheduler.config)


# Inference is ran for every server call
# Reference preloaded global pipeline here. 
def inference(model_inputs:dict) -> dict:
    global model
    # Parse pipeline arguments
    prompt = model_inputs.get('prompt', None)
    num_inference_steps = model_inputs.get('num_inference_steps', 20)
    image_guidance_scale=model_inputs.get('image_guidance_scale', 2)
    guidance_scale=model_inputs.get('prompt_guidance_scale', 7)
    just_test_img=model_inputs.get('test_mode', False)
    # num_images_per_prompt=model_inputs.get('num_images_per_prompt', 1)

    # decode image
    base64_string = model_inputs.get('image')
    image = dataUrlToPil(base64_string)

    if prompt == None:
        return {'message': "No prompt provided"}

    if just_test_img:
        # just test sending & getting back images
        edited_image = image

    else:
        # Run the model
        edited_image = model(prompt, image=image, 
            num_inference_steps=num_inference_steps, 
            image_guidance_scale=image_guidance_scale, 
            guidance_scale=guidance_scale).images[0]

    # Return the results as a dictionary
    out = {"imageUrl": pilToDataUrl(edited_image),
           "prompt": prompt,
           "guidance_scale": guidance_scale,
           "prompt_guidance_scale": image_guidance_scale,
           "test_mode": just_test_img
        }
    
    return out