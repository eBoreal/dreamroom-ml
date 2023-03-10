# This file is used to verify your http server acts as expected
# Run it with `python3 test.py``

import requests
import base64

  
from utils import stringToPil


# params
test_img_path = "data/input/venus-of-milo-512.jpg"
prompt = "turn her into a cyborg"
num_inference_steps=20
image_guidance_scale=1.4
prompt_guidance_scale=7
num_images_per_prompt= 1
test_mode=False

test_name = f"test-venus-of-milo"


def make_test_example():
    # convert img to base64
    with open(test_img_path, "rb") as image2string:
        base64_bytes  = base64.b64encode(image2string.read())

    # pass it as string for json
    base64_string = base64_bytes.decode('utf-8')

    model_inputs = {'prompt': prompt,
                    'image': base64_string,
                    'num_inference_steps': num_inference_steps,
                    'image_cfg_scale': image_guidance_scale,
                    'text_cfg_scale': prompt_guidance_scale,
                    'num_images_per_prompt': num_images_per_prompt,
                    'test_mode': test_mode
                    }
    return model_inputs

def check_server_status():
    check = requests.get("http://localhost:8000/healthcheck/")
    return check

def test_inference():
    
    model_inputs = make_test_example()

    response = requests.post('http://localhost:8000/', json = model_inputs, 
    )
    print("Server Inference with: ", response.status_code)

    # save responnse
    output = response.json()
    for idx, image in enumerate(output): 
        print("Saving images")

        txt_cfg = image["text_cfg_scale"]
        img_cfg = image["image_cfg_scale"]
        steps = image["steps"]

        img_path = f"data/output/{test_name}-{txt_cfg}-{img_cfg}-{steps}-{idx}.jpeg"
        print("Saving: ", img_path)

        img = stringToPil(image['image'])
        img.save(img_path)


if __name__ == "__main__":

    res = check_server_status()
    server_health = res.json()

    if server_health["state"] == 'healthy':
        try:
            print("Testing inference")
            test_inference()
        except Exception as e:
            print(server_health)
            print(e)

    else: 
        print(server_health)