
# 🍌 Serverless pix2pix

Ready to deploy inference endpoint for pix2pix. 

# Disclaimer

This repo uses the HuggingFace diffusers' implementation of Tim Brooks et al. Instruct Pix2pix model - https://www.timothybrooks.com/instruct-pix2pix


# How to interact with the service

## Run the server
2 options:

Without docker
```
pip install -r requirements.txt
python3 server.py
```

Or with docker

```
docker build -t pix2pix .
docker run -p 8000:8000 pix2pix
```

## Model Inputs

The model accepts the following inputs:

* `prompt` (str, required)
* `image` (base64 str, required) - A base64 string of the image (data:image/type;base64,.... also accepeted) should be 512x512 or another standard Stable Diffusion 1.5 resolution for best results
* `seed` (int, optional, defaults to 42)
* `text_cfg_scale` (float, optional, default 7)
* `image_cfg_scale` (float, optional, default 1.5)
* `steps` (int, optional, default to 20)
* `randomize_cfg` (boolean, optional, default False)
* `randomize_seed` (boolean, optional, default True)
* `image_cfg_scale` (float, optional, default 1.5)

Additional parameters:
* `test_mode` (boolean, optional, default False)
* `toDataUrl` (boolean, optional, default False) - if you want output "data:image/type;base64,...."


Not implemented
* `negative_prompt`
* `num_images_per_prompt`



## Model Output

The model outputs:

A list of image objects where each has the following properties:
* `image` (base64 str) - base64 or base64 with data_url prefix if specified
* `seed` (int)
* `text_cfg_scale` (float)
* `image_cfg_scale` (float)
* `steps` (int)



# Example

- Checkout the test.py for an example



# Examples of generated images

Venus de Milo             |  Turn her into a cyborg
:-------------------------:|:-------------------------:
![](https://github.com/eBoreal/serverless-pix2pix/blob/main/data/input/venus-of-milo-512.jpg)  |  ![](https://github.com/eBoreal/serverless-pix2pix/blob/main/data/output/venus-of-milo-512.jpeg) 

<br>

Elon            |  Turn him into a cyborg
:-------------------------:|:-------------------------:
![](https://github.com/eBoreal/serverless-pix2pix/blob/main/data/input/elon-512.jpg) |  ![](https://github.com/eBoreal/serverless-pix2pix/blob/main/data/output/elon-2-512.jpeg)

<br>

# Helpful Links

Learn more about Instruct Pix2Pix here - https://www.timothybrooks.com/instruct-pix2pix

And Hugging Face support there - https://huggingface.co/timbrooks/instruct-pix2pix

Understand the 🍌 [Serverless framework](https://docs.banana.dev/banana-docs/core-concepts/inference-server/serverless-framework) and functionality of each file within it.

<br>

## Use Banana for scale.
