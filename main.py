from model import ImagePreprocessor, OnnxClassifier
import base64
from PIL import Image
from io import BytesIO

preprocessor = ImagePreprocessor()
classifier = OnnxClassifier()

def run(request):
    image_b64 = request.get("image_base64")
    if not image_b64:
        return {"error": "image_base64 key is missing"}

    image_bytes = base64.b64decode(image_b64)
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    input_array = preprocessor.run(img)
    pred_class = classifier.predict(input_array)

    return {"class_index": pred_class}
