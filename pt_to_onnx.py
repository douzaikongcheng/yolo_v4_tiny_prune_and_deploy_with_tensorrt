from yolo import YOLO
import torch
from torch.autograd import Variable
import onnx
print(torch.__version__)
import argparse

parser = argparse.ArgumentParser("test")
parser.add_argument('--test_mode', type=str, default='camera', help='location of the data corpus')
parser.add_argument('--prune', action='store_true', default=True, help='use auxiliary tower')
args = parser.parse_args()

input_name = ['input']
output_name = ['output']
batch_size = 1
input = Variable(torch.randn(batch_size, 3, 256, 256), requires_grad=False).cuda()
model = YOLO(args).prune().cuda()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
state_dict = torch.load('model_data/finetune_prune_model.pth', map_location=device)
model.load_state_dict(state_dict)

torch.onnx.export(model, input, 'yolo_v4_tiny_prune.onnx', input_names=input_name, output_names=output_name, verbose=True, opset_version=11)

test = onnx.load('yolo_v4_tiny_prune.onnx')
onnx.checker.check_model(test)
print("==> Passed")
