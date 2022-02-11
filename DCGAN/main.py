
from config import *
from data import get_celeba
from model import weights_init, Generator, Discriminator
from train import train
from generate import generate
from config import *

def main():
    dataloader = get_celeba(params)

    netG = Generator(params).to(device)
    netG.apply(weights_init)

    netD = Discriminator(params).to(device)
    netD.apply(weights_init)

    optimizerD = optim.Adam(netD.parameters(), lr=params['lr'], betas=(params['beta1'], 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=params['lr'], betas=(params['beta1'], 0.999))

    for epoch in range(params['nepochs']):
        train(netD, netG, optimizerD, optimizerG, dataloader, epoch)

    generate(1, netG)

if __name__ == '__main__':
    main()