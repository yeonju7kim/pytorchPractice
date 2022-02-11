
from config import *
from data import get_celeba
from model import weights_init, Generator, Discriminator
from train import train
from generate import *
from config import *

def main():
    dataloader = get_celeba(params)

    netG = Generator(params).to(device)
    netG.apply(weights_init)

    netD = Discriminator(params).to(device)
    netD.apply(weights_init)

    optimizerD = optim_D(netD.parameters(), lr=params['lr'], betas=(params['beta1'], 0.999))
    optimizerG = optim_G(netG.parameters(), lr=params['lr'], betas=(params['beta1'], 0.999))

    for epoch in range(params['nepochs']):
        train(netD, netG, optimizerD, optimizerG, dataloader, epoch)
        # pass
    generate(1, netG)
    # generate(2, load_path= "checkpoint/model_epoch_0.pth")
    # generate_gradation_with_interpolation(2, load_path= "checkpoint/model_epoch_1.pth")

if __name__ == '__main__':
    main()