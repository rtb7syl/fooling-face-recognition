import onnx
from onnx2pytorch import ConvertModel

path_to_onnx_model='face_recognition/insightface_torch/model.onnx'
onnx_model = onnx.load(path_to_onnx_model)
pytorch_model = ConvertModel(onnx_model)
pytorch_model.save('face_recognition/insightface_torch/weights/ms1mv3_retinaface_resnet18.pth')