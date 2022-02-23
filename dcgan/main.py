from model import *
from train import train
from generate import *
from config import *
from utils.history_manage import *

def main():
    '''
    마지막으로 autosaved된 model 불러온다. Generator랑 Discriminator를 불러온다.
    gan은 최적의 모델을 정할 수 있는 기준이 없다. 그래서 일단은 마지막으로 autosaved된 모델을 불러왔다.
    '''
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

    # optimizer 선언
    optimizerD = optim_D(netD.parameters(), lr=params['lr'], betas=(params['beta1'], params['beta2']))
    optimizerG = optim_G(netG.parameters(), lr=params['lr'], betas=(params['beta1'], params['beta2']))

    # history는 train 할 때 계산되는 error를 csv파일로 저장하기 위해 만들었다.
    history = ValueHistory()

    # celebA 데이터셋을 불러온다.
    dataloader = get_celeba(params)
    for epoch in range(params['nepochs']):
        train(netD, netG, optimizerD, optimizerG, dataloader, history, epoch, last_epoch + 1)
        # 가장 최적의 모델을 알 수 없으니 일단 모든 epoch의 모델을 저장했다.
        # history.save_csv_all_history(f"train_err_{epoch + last_epoch + 1}", "../history")
    history.save_csv_all_history(f"train_err_overall", "../history")

    # 학습한 generator로 1개의 random으로 생성된 latent vector로 1개의 이미지를 합성
    generate(1, netG)
    # generate(2, load_path= "checkpoint/model_epoch_0.pth")
    # generate_gradation_with_interpolation(2, load_path= "checkpoint/model_epoch_1.pth")
    # generate_gradation_with_interpolation(10)
    # 학습한 generator로 10개의 random으로 생성된 latent vector에 interpolation을 추가한 후 이미지를 합성
    generate_gradation_with_interpolation(10, netG)
    # 학습한 generator로 30개의 random으로 생성된 latent vector를 더하고 빼서 새로운 letent vector를 만든 다음 10개 이미지를 합성
    generate_with_calculated_context_vector(10, netG)

if __name__ == '__main__':
    main()

