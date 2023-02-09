from PIL import Image
from io import BytesIO
import base64

def stringToPil(
    img_string: str
    ):
    is_data_url = True if len(img_string.split(",")) > 0 else False
    
    if is_data_url:
        base64string = img_string.split(",")[1]
    else:
        base64string = img_string

    img = Image.open(BytesIO(base64.b64decode(base64string,
         validate=True))).convert("RGB")
    return img


def pilToString(
    img: Image, 
    dataUrl=False
    ):
    
    im_file = BytesIO()
    img.save(im_file, format="JPEG")
    im_bytes = im_file.getvalue()

    base64string =  base64.b64encode(im_bytes).decode('utf-8')
    if dataUrl:
        return 'data:image/jpeg;base64,' + base64string
    else:
        return base64string