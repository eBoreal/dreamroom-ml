# This file is used to verify your http server acts as expected
# Run it with `python3 test.py``

import requests
import base64
import requests
import base64

  


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
    # imageString = imageStringToPil

# pass it as string for json
base64_string = base64_bytes.decode('utf-8')

model_inputs = {'prompt': prompt,
                'imageString': base64_string,
                'num_inference_steps': num_inference_steps,
                'image_guidance_scale': image_guidance_scale,
                'prompt_guidance_scale': prompt_guidance_scale,
                'num_images_per_prompt': num_images_per_prompt
                }

res = requests.post('http://localhost:8000', json = model_inputs, 
)

print("responded with: ", res.status_code)

# save responnse
output = res.json()
print(output)
