import os
import torch
from model import ResNet
from data import *
from config import *
from torch import optim, tensor
import torchvision.transforms as transforms
from resnet import RandomResize


model = ResNet.from_pretrained('resnet18', num_classes=10).to(device)
model = ResNet.from_name('resnet18', override_params={"num_classes": 10}).to(device)
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0002)
# TODO 0.1이고 err가 변하지 않을 때 10씩 나눔. 구현 필요



transform = transforms.Compose([
    RandomResize((transform_param['short_resize'], transform_param['long_resize'] + 1)),
    transforms.RandomCrop((transform_param['random_crop'], transform_param['random_crop'])),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # TODO 0~255로 사용한 것으로 보이는데,에러가 나서 일단 0~1로
    transforms.Normalize(transform_param['pixel_mean'], transform_param['pixel_std'])
    # TODO pixel mean 을 뺐다고 하는데 값을 모름..
])

train_dataloader, test_dataloader = get_cifar10_dataloader(batchsize, transform=transform)
file_name = 'resnet18_cifar10'

def main():
    for i in range(iteration):
        train_acc = train(i)
        test_acc = test(i)
        save(i, train_acc, test_acc)


def train(epoch):
    print(f'[ Train epoch: {epoch} ]')
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets, targets_onehot) in enumerate(train_dataloader):
        inputs, targets, targets_onehot = inputs.to(device), targets.to(device), tensor(targets_onehot).to(device)
        optimizer.zero_grad()

        outputs = model(inputs.float())
        loss = criterion(outputs, targets_onehot)
        loss.backward()

        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)

        total += targets.size(0)
        current_correct = predicted.eq(targets).sum().item()
        correct += current_correct

        if batch_idx % 100 == 0:
            print('\nCurrent batch:', str(batch_idx))
            print('Current batch average train accuracy:', current_correct / targets.size(0))
            print('Current batch average train loss:', loss.item() / targets.size(0))

    print('\nTotal average train accuarcy:', correct / total)
    print('Total average train loss:', train_loss / total)
    return correct / total

def test(epoch):
    print(f'[ Test epoch: {epoch} ]')
    model.eval()
    loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets, targets_onehot) in enumerate(test_dataloader):
        inputs, targets, targets_onehot = inputs.to(device), targets.to(device), tensor(targets_onehot).to(device)
        total += targets.size(0)

        outputs = model(inputs.float())
        loss += criterion(outputs, targets_onehot).item()

        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()

    print('\nTotal average test accuarcy:', correct / total)
    print('Total average test loss:', loss / total)

    return correct / total

def save(iter, train_acc, test_acc):
    state = {
        'net': model.state_dict()
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/' + file_name + f'_{iter}_{train_acc}_{test_acc}.pth')
    print('Model Saved!')

if __name__ == '__main__':
    main()
