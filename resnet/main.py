import torch
from resnet import ResNet
from torch.utils.data import DataLoader
from cifar10_data_loader import *
from torch import cuda, nn, optim, tensor

batchsize = 20
iteration = 5
device = 'cuda' if cuda.is_available() else 'cpu'
learning_rate = 0.1
criterion = nn.CrossEntropyLoss()
model = ResNet.from_pretrained('resnet18', num_classes=10).to(device)
# model = ResNet.from_name('resnet18', override_params={"num_classes": 10}).to(device)
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0002)
train_dataloader, test_dataloader = get_cifar10_dataloader(batchsize)
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