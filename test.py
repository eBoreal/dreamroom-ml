# This file is used to verify your http server acts as expected
# Run it with `python3 test.py``

import requests
import base64
  
  

  
# convert img to base64
with open("scratch.jpg", "rb") as image2string:
    img = base64.b64encode(image2string.read())

model_inputs = {'prompt': 'Make the background red',
                'image': str(img)}

res = requests.post('http://localhost:8000', json = model_inputs)

print(res.json())