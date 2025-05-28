import torch
from pytorch_model import Classifier
from wrapper import PreprocessWrapper

# def export_to_onnx():
#     model = Classifier()
#     model.load_state_dict(torch.load("pytorch_model_weights.pth"))
#     model.eval()

#     dummy_input = torch.randn(1, 3, 224, 224)

#     torch.onnx.export(
#         model,
#         dummy_input,
#         "classifier.onnx",
#         input_names=["input"],
#         output_names=["output"],
#         dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
#         opset_version=11
#     )

#     print("Exported ONNX successfully")

def export_to_onnx():
    model = Classifier()
    model.load_state_dict(torch.load("pytorch_model_weights.pth"))
    model.eval()

    wrapped = PreprocessWrapper(model)

    dummy_input = torch.randint(0, 255, (1, 224, 224, 3), dtype=torch.uint8)  # Simulate raw image

    torch.onnx.export(
        wrapped,
        dummy_input,
        "classifier_preproc.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=11
    )
    print("✅ Exported ONNX with preprocessing included!")


if __name__ == "__main__":
    export_to_onnx()
