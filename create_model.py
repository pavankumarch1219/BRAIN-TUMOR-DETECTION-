import torch
from torch.nn import Sequential, Linear, SELU, Dropout, LogSigmoid
from torchvision.models import resnet50
import os

os.makedirs("models", exist_ok=True)

model = resnet50(pretrained=True)

n_inputs = model.fc.in_features

model.fc = Sequential(
    Linear(n_inputs, 2048),
    SELU(),
    Dropout(p=0.4),
    Linear(2048, 2048),
    SELU(),
    Dropout(p=0.4),
    Linear(2048, 4),
    LogSigmoid()
)

torch.save(model.state_dict(), "models/bt_resnet50_model.pt")

print("Model created successfully")
