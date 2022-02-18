import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from model import Generator, Discriminator
from config import *
from data import *
from PIL import Image
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

def generate_with_calculated_context_vector(num_output, netG=None, load_path=None):
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

    noise = torch.randn(int(num_output) * 3, params['nz'], 1, 1, device=device)

    noise_list = []
    new_noise = []
    for i in range(int(num_output)):
        noise_list.append(noise[3 * i])
        noise_list.append(noise[3 * i + 1])
        noise_list.append(noise[3 * i + 2])
        new_noise.append(noise[3 * i] + noise[3 * i + 1] - noise[3 * i + 2])
    noise_tensor = torch.stack(noise_list, dim=0)
    new_noise_tensor = torch.stack(new_noise, dim=0)
    with torch.no_grad():
        generated_img = netG(noise_tensor).detach().cpu()
        generated_new_img = netG(new_noise_tensor).detach().cpu()

    if os.path.exists("result") == False:
        os.mkdir("result")
    for i in range(int(num_output)):
        result = [generated_img[3 * i], generated_img[3 * i + 1], generated_img[3 * i + 2], generated_new_img[i]]
        result_img = np.array(np.transpose(vutils.make_grid(result, padding=2, normalize=True), (1, 2, 0))) * 255
        result_img = result_img.astype(np.uint8)
        im = Image.fromarray(result_img)
        im.save(f"result/result{i}.jpeg")

# def generate_with_calculated_context_vector(num_output, netD=None, netG=None, load_path=None):
#     if netD == None or netG == None:
#         if load_path is None :
#             last_model, last_epoch = get_last_model()
#             print(f"use {last_epoch} epoch")
#             netG = Generator(params).to(device)
#             netD = Discriminator(params).to(device)
#             netG.load_state_dict(last_model['generator'])
#             netD.load_state_dict(last_model['discriminator'])
#         else:
#             state_dict = torch.load(load_path)
#             netG = Generator(params).to(device)
#             netD = Discriminator(params).to(device)
#             netG.load_state_dict(state_dict['generator'])
#             netD.load_state_dict(state_dict['discriminator'])
#
#     dataloader = get_celeba(params)
#     abspath_dictionary = get_abspath_dictionary(dataloader)
#
#     attr_dictionary = read_attr_label()
#
#     filename_list_glasses_man = get_abspath_list(get_image_list_by_common_attr(['Eyeglasses', 'Male'],[], attr_dictionary, num_output), abspath_dictionary)
#     filename_list_not_glasses_man = get_abspath_list(get_image_list_by_common_attr(['Male'], ['Eyeglasses'], attr_dictionary, num_output), abspath_dictionary)
#     filename_list_not_glasses_woman = get_abspath_list(get_image_list_by_common_attr([], ['Male', 'Eyeglasses'], attr_dictionary, num_output), abspath_dictionary)
#
#     img_list_glasses_man = imread_multiple_to_tensor(filename_list_glasses_man).float()
#     img_list_not_glasses_man = imread_multiple_to_tensor(filename_list_not_glasses_man).float()
#     img_list_not_glasses_woman = imread_multiple_to_tensor(filename_list_not_glasses_woman).float()
#
#     vector_list_glass_man = netD(img_list_glasses_man).view(-1)
#     vector_list_not_glass_man = netD(img_list_not_glasses_man).view(-1)
#     vector_list_not_glass_woman = netD(img_list_not_glasses_woman).view(-1)
#
#     vector_generator = vector_list_glass_man - vector_list_not_glass_man + vector_list_not_glass_woman
#
#     # noise = torch.randn(int(num_output), params['nz'], 1, 1, device=device)
#     #
#     # noise_list = []
#     # for n in noise:
#     #     for i in range(10):
#     #         noise_list.append(n)
#     #
#     #
#     # interpolation = torch.Tensor(np.arange(-1, 1, 0.2))
#     # for i in range(len(noise)):
#     #     for j in range(10):
#     #         noise_list[10 * i + j] = noise_list[10 * i + j] + interpolation[j]
#     #
#     # noise_10 = torch.stack(noise_list, dim=0)
#
#     with torch.no_grad():
#         generated_img = netG(vector_generator).detach().cpu()
#
#     plt.axis("off")
#     plt.title("Generated Images")
#     plt.imshow(np.transpose(vutils.make_grid(generated_img, padding=2, normalize=True), (1, 2, 0)))
#
#     plt.show()