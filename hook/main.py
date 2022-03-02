# REFERENCE : https://github.com/cosmic-cortex/pytorch-hooks-tutorial/blob/master/hooks.ipynb

import torch
from torchvision.models import resnet34

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = resnet34(pretrained=True)
model = model.to(device)

class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)

    def clear(self):
        self.outputs = []

save_output = SaveOutput()

hook_handles = []

# NOTE : layer를 돌며 Conv2d인 layer에 hook 건다.

for layer in model.modules():
    if isinstance(layer, torch.nn.modules.conv.Conv2d):
        handle = layer.register_forward_hook(save_output)
        hook_handles.append(handle)

from PIL import Image
from torchvision import transforms as T

image = Image.open('cat.jpg')
transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
X = transform(image).unsqueeze(dim=0).to(device)

out = model(X)

import matplotlib.pyplot as plt

def module_output_to_numpy(tensor):
    return tensor.detach().to('cpu').numpy()

images = module_output_to_numpy(save_output.outputs[0])

with plt.style.context("seaborn-white"):
    plt.figure(figsize=(20, 20), frameon=False)
    for idx in range(16):
        plt.subplot(4, 4, idx+1)
        plt.imshow(images[0, idx])
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plt.show()