from model import ImagePreprocessor, OnnxClassifier
import numpy as np

def test_preprocessing():
    print("Testing image preprocessing...")
    preprocessor = ImagePreprocessor()
    input_array = preprocessor.run("n01667114_mud_turtle.JPEG")
    
    assert input_array.shape == (1, 3, 224, 224), f"Unexpected input shape: {input_array.shape}"
    assert input_array.dtype == np.float32, f"Input dtype must be float32, got {input_array.dtype}"
    print("Image preprocessing passed.\n")
    return input_array

def test_model_loading():
    print("Testing ONNX model loading...")
    try:
        classifier = OnnxClassifier()
        print("Model loaded successfully.\n")
        return classifier
    except Exception as e:
        raise RuntimeError(f"Model loading failed: {e}")

def test_inference(classifier, input_array):
    print("Running inference...")
    prediction = classifier.predict(input_array)
    
    assert isinstance(prediction, int), f"Prediction should be int, got {type(prediction)}"
    print("Inference successful.")
    print(f"Predicted Class Index: {prediction}\n")

def test_all():
    print("\nStarting ML deployment test suite...\n")
    input_array = test_preprocessing()
    classifier = test_model_loading()
    test_inference(classifier, input_array)
    print("All tests passed!\n")

if __name__ == "__main__":
    test_all()