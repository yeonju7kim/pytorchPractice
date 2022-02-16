import torch
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from model import Generator, Discriminator
from config import *
from data import *
def generate(num_output, netG = None, load_path = None):
    state_dict = torch.load(load_path)

    params = state_dict['params']

    if netG == None:
        netG = Generator(params).to(device)
        netG.load_state_dict(state_dict['generator'])

    noise = torch.randn(int(num_output), params['nz'], 1, 1, device=device)

    with torch.no_grad():
        generated_img = netG(noise).detach().cpu()

    plt.axis("off")
    plt.title("Generated Images")
    plt.imshow(np.transpose(vutils.make_grid(generated_img, padding=2, normalize=True), (1,2,0)))

    plt.show()


def generate_gradation_with_interpolation(num_output, netG=None, load_path=None):
    if netG == None:
        if load_path is None :
            last_model, last_epoch = get_last_model()
            print(f"use {last_epoch} epoch")
            netG = Generator(params).to(device)
            netG.load_state_dict(last_model['generator'])
        else:
            state_dict = torch.load(load_path)
            netG = Generator(params).to(device)
            netG.load_state_dict(state_dict['generator'])

    noise = torch.randn(int(num_output), params['nz'], 1, 1, device=device)

    noise_list = []
    for n in noise:
        for i in range(10):
            noise_list.append(n)


    interpolation = torch.Tensor(np.arange(-1, 1, 0.2))
    for i in range(len(noise)):
        for j in range(10):
            noise_list[10 * i + j] = noise_list[10 * i + j] + interpolation[j]

    noise_10 = torch.stack(noise_list, dim=0)

    with torch.no_grad():
        generated_img = netG(noise_10).detach().cpu()

    plt.axis("off")
    plt.title("Generated Images")
    plt.imshow(np.transpose(vutils.make_grid(generated_img, padding=2, normalize=True), (1, 2, 0)))

    plt.show()


def generate_with_calculated_context_vector(num_output, netD=None, netG=None, load_path=None):
    if netD == None or netG == None:
        if load_path is None :
            last_model, last_epoch = get_last_model()
            print(f"use {last_epoch} epoch")
            netG = Generator(params).to(device)
            netD = Discriminator(params).to(device)
            netG.load_state_dict(last_model['generator'])
            netD.load_state_dict(last_model['discriminator'])
        else:
            state_dict = torch.load(load_path)
            netG = Generator(params).to(device)
            netD = Discriminator(params).to(device)
            netG.load_state_dict(state_dict['generator'])
            netD.load_state_dict(state_dict['discriminator'])

    img_list_glass_man = get_image_list_by_common_attr(['Eyeglasses', 'Male'], num_output)
    img_list_not_glass_man = get_image_list_by_common_attr(['Male'], ['Eyeglasses'], num_output)
    img_list_not_glass_woman = get_image_list_by_common_attr([], ['Male', 'Eyeglasses'], num_output)

    vector_list_glass_man = netD(img_list_glass_man)
    vector_list_not_glass_man = netD(img_list_not_glass_man)
    vector_list_not_glass_woman = netD(img_list_not_glass_woman)

    vector_generator = vector_list_glass_man - vector_list_not_glass_man + vector_list_not_glass_woman

    # noise = torch.randn(int(num_output), params['nz'], 1, 1, device=device)
    #
    # noise_list = []
    # for n in noise:
    #     for i in range(10):
    #         noise_list.append(n)
    #
    #
    # interpolation = torch.Tensor(np.arange(-1, 1, 0.2))
    # for i in range(len(noise)):
    #     for j in range(10):
    #         noise_list[10 * i + j] = noise_list[10 * i + j] + interpolation[j]
    #
    # noise_10 = torch.stack(noise_list, dim=0)

    with torch.no_grad():
        generated_img = netG(vector_generator).detach().cpu()

    plt.axis("off")
    plt.title("Generated Images")
    plt.imshow(np.transpose(vutils.make_grid(generated_img, padding=2, normalize=True), (1, 2, 0)))

    plt.show()