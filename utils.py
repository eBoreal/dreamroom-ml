from PIL import Image
from io import BytesIO
import base64

def string_to_pil(img_string):
    return Image.open(BytesIO(base64.b64decode(img_string,
         validate=True)))

def pil_to_string(img):
        im_file = BytesIO()
        img.save(im_file, format="JPEG")
        im_bytes = im_file.getvalue()
        return base64.b64encode(im_bytes).decode('utf-8')