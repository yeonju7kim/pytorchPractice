import os.path
from config import *


def train(netD, netG, optimizerD, optimizerG, dataloader, history, epoch, last_epoch):
    iters = 0
    errD_losses = 0
    errG_losses = 0
    D_x_losses = 0
    D_G_z1_losses = 0
    D_G_z2_losses = 0

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
        label.fill_(fake_label)
        output = netD(fake_data.detach()).view(-1)
        # fake_data(Tensor) 와 관련된 모든 operation과 tensor는 tracking 되기 때문에 이것을 detach함
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item() # fake 이미지를 fake로 추론

        errD = errD_real + errD_fake
        optimizerD.step()

        netG.zero_grad()
        label.fill_(real_label)
        output = netD(fake_data).view(-1)
        errG = criterion(output, label)
        errG.backward()

        D_G_z2 = output.mean().item() # real로 추론
        optimizerG.step()

        if i%50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch + last_epoch, params['nepochs'] + last_epoch, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # history.add_history("d error", errD.item())
        # history.add_history("g error", errG.item())
        # history.add_history("average output of real data", D_x)
        # history.add_history("fake data output before d update", D_G_z1)
        # history.add_history("fake data output after d update", D_G_z2)

        if (iters % 100 == 0) or ((epoch == params['nepochs']-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake_data = netG(fixed_noise).detach().cpu()

        iters += 1

        errD_losses += errD.item()
        errG_losses += errG.item()
        D_x_losses = D_x
        D_G_z1_losses = D_G_z1
        D_G_z2_losses = D_G_z2

    save(netG, netD, optimizerG, optimizerD, epoch + last_epoch + 1)

    dataloader_length = len(dataloader)
    history.add_history("d error", errD_losses/ dataloader_length)
    history.add_history("g error", errG_losses/ dataloader_length)
    history.add_history("average output of real data", D_x_losses/ dataloader_length)
    history.add_history("fake data output before d update", D_G_z1_losses/ dataloader_length)
    history.add_history("fake data output after d update", D_G_z2_losses/ dataloader_length)

def save(netG, netD, optimizerG, optimizerD, epoch):
    if os.path.exists("checkpoint") == False:
        os.mkdir("checkpoint")
    torch.save({
        'generator' : netG.state_dict(),
        'discriminator' : netD.state_dict(),
        'optimizerG' : optimizerG.state_dict(),
        'optimizerD' : optimizerD.state_dict(),
        'params' : params
        }, 'checkpoint/model_epoch_{:03d}.pth'.format(epoch))
