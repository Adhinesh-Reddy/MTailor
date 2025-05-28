import onnxruntime as ort
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


class ImagePreprocessor:
    def __init__(self):
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def run(self, img: Image.Image) -> np.ndarray:
        tensor = self.preprocess(img).unsqueeze(0)
        return tensor.numpy()

class OnnxClassifier:
    def __init__(self, model_path="classifier.onnx"):
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, input_array: np.ndarray) -> int:
        outputs = self.session.run([self.output_name], {self.input_name: input_array})
        return int(np.argmax(outputs[0]))

if __name__ == "__main__":
    preprocessor = ImagePreprocessor()
    image_array = preprocessor.run("n01667114_mud_turtle.JPEG")

    classifier = OnnxClassifier()
    pred_class = classifier.predict(image_array)

    print("Predicted Class Index:", pred_class)