import torchvision.utils as vutils
from config import *

def train(netD, netG, optimizerD, optimizerG, dataloader, epoch):
    iters = 0
    for i, data in enumerate(dataloader, 0):
        real_data = data[0].to(device)
        b_size = real_data.size(0)

        netD.zero_grad()
        label = torch.full((b_size, ), real_label, device=device).float()
        output = netD(real_data).view(-1)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        noise = torch.randn(b_size, params['nz'], 1, 1, device=device)
        fake_data = netG(noise)
        label.fill_(fake_label  )
        output = netD(fake_data.detach()).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()

        errD = errD_real + errD_fake
        optimizerD.step()

        netG.zero_grad()
        label.fill_(real_label)
        output = netD(fake_data).view(-1)
        errG = criterion(output, label)
        errG.backward()

        D_G_z2 = output.mean().item()
        optimizerG.step()

        if i%50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, params['nepochs'], i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        if (iters % 100 == 0) or ((epoch == params['nepochs']-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake_data = netG(fixed_noise).detach().cpu()

        iters += 1

    save(netG, netD, optimizerG, optimizerD, epoch)

def save(netG, netD, optimizerG, optimizerD, epoch):
    torch.save({
        'generator' : netG.state_dict(),
        'discriminator' : netD.state_dict(),
        'optimizerG' : optimizerG.state_dict(),
        'optimizerD' : optimizerD.state_dict(),
        'params' : params
        }, 'checkpoint/model_epoch_{}.pth'.format(epoch))