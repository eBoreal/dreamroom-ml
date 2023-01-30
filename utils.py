from PIL import Image
from io import BytesIO
import base64

def dataUrlToPil(img_string):
    base64string = img_string.split(",")[1]
    return Image.open(BytesIO(base64.b64decode(base64string,
         validate=True)))

def pilToDataUrl(img):
        im_file = BytesIO()
        img.save(im_file, format="JPEG")
        im_bytes = im_file.getvalue()
        return 'data:image/jpeg;base64,' + base64.b64encode(im_bytes).decode('utf-8')