
from config import *
from data import *
from model import weights_init, Generator, Discriminator
from train import train
from generate import *
from config import *
import torch.utils.model_zoo as model_zoo
import os, re
from utils.history import *

def main():
    dataloader = get_celeba(params)

    loaded_model, last_epoch = get_last_model()

    netG = Generator(params).to(device)

    if loaded_model == None:
        netG.apply(weights_init)
    else:
        netG.load_state_dict(loaded_model['generator'])

    netD = Discriminator(params).to(device)

    if loaded_model == None:
        netD.apply(weights_init)
    else:
        netD.load_state_dict(loaded_model['discriminator'])

    optimizerD = optim_D(netD.parameters(), lr=params['lr'], betas=(params['beta1'], 0.999))
    optimizerG = optim_G(netG.parameters(), lr=params['lr'], betas=(params['beta1'], 0.999))


    history = ValueHistory()
    for epoch in range(params['nepochs']):
        # train(netD, netG, optimizerD, optimizerG, dataloader, history, epoch, last_epoch + 1)
        history.save_csv_all_history(f"train_err_{epoch + last_epoch + 1}", "../history")
        # pass
    # generate(1, netG)
    # generate(2, load_path= "checkpoint/model_epoch_0.pth")
    # generate_gradation_with_interpolation(2, load_path= "checkpoint/model_epoch_1.pth")
    # generate_gradation_with_interpolation(10)
    # generate_gradation_with_interpolation(10, netG)
    generate_with_calculated_context_vector(3, netD, netG)

if __name__ == '__main__':
    main()

