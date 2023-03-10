
import os
import banana_dev as banana
from dotenv import load_dotenv

import requests
import base64
from utils import stringToPil


# params
test_img_path = "data/input/venus-of-milo-512.jpg"
prompt = "turn her into a cyborg"
num_inference_steps=20
image_guidance_scale=1.5
prompt_guidance_scale=10
num_images_per_prompt= 1

test_name = f"venus-of-milo-{num_inference_steps}-{image_guidance_scale}-{prompt_guidance_scale}"

# convert img to base64
with open(test_img_path, "rb") as image2string:
    base64_bytes  = base64.b64encode(image2string.read())

# pass it as string for json
base64_string = base64_bytes.decode('utf-8')

model_inputs = {'prompt': prompt,
                'image': base64_string,
                'num_inference_steps': num_inference_steps,
                'image_guidance_scale': image_guidance_scale,
                'prompt_guidance_scale': prompt_guidance_scale,
                'num_images_per_prompt': num_images_per_prompt
                }

load_dotenv()
api_key = os.getenv("API_KEY")
model_key = os.getenv("MODEL_KEY")

# for logging
inputs_to_log = model_inputs
inputs_to_log["image"] = model_inputs["image"][:20]
print("Sending request with: ", inputs_to_log)


# Run the model
try:
    call_id = "abc"
    check = banana.check(api_key, call_id)
    print("check responded with: ", check.status_code)
    checkRes = check.json()
    print(checkRes)
except Exception as e:
    print(e)
    print("check failed")

response = banana.run(api_key, model_key, model_inputs)


print("responded with: ", response.status_code)

# save responnse
output = response.json()
for idx, res in enumerate(response['modelOutputs']): 
    print(output.keys())
    img = stringToPil(output[0]['image-0'])
    img.save(f"data/output/{test_name}-{idx}.jpeg")
    print("Saving result under: ", "data/output/{test_name}-{idx}.jpeg")
