import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from model import Generator
from config import *

def generate(num_output, netG = None, load_path = None):
    state_dict = torch.load(load_path)

    params = state_dict['params']

    if netG == None:
        netG = Generator(params).to(device)
        netG.load_state_dict(state_dict['generator'])

    noise = torch.randn(int(num_output), params['nz'], 1, 1, device=device)

    # Turn off gradient calculation to speed up the process.
    with torch.no_grad():
        generated_img = netG(noise).detach().cpu()

    plt.axis("off")
    plt.title("Generated Images")
    plt.imshow(np.transpose(vutils.make_grid(generated_img, padding=2, normalize=True), (1,2,0)))

    plt.show()