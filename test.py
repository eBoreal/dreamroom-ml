# This file is used to verify your http server acts as expected
# Run it with `python3 test.py``

import requests
import base64
import requests
from PIL import Image
from io import BytesIO
import base64

  
from utils.img_helpers import string_to_pil, pil_to_string


  
# convert img to base64
with open("scratch.jpg", "rb") as image2string:
    base64_bytes  = base64.b64encode(image2string.read())

# pass it as string for json
base64_string = base64_bytes.decode('utf-8')

model_inputs = {'prompt': 'Make the background red',
                'image': base64_string}

res = requests.post('http://localhost:8000', json = model_inputs, 
)

#     files = {'image': img}

output = res.json()

img = string_to_pil(output['image'])
img.save("result.jpeg")

