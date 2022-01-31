import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import torchvision
from matplotlib.pyplot import imshow

from tutorials.Load import load_cifar_10_data


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # 배치를 제외한 모든 차원을 평탄화(flatten)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

train_data, train_filenames, train_labels, test_data, test_filenames, test_labels, label_names = load_cifar_10_data('../data/cifar-10-batches-py')

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

for epoch in range(2):   # 데이터셋을 수차례 반복합니다.
    running_loss = 0.0

    for i in range(len(train_data)):
        # [inputs, labels]의 목록인 data로부터 입력을 받은 후;
        inputs, labels = train_data[i : i + 1], torch.from_numpy(train_labels[i : i + 1]).long()

        # 변화도(Gradient) 매개변수를 0으로 만들고
        optimizer.zero_grad()

        # 순전파 + 역전파 + 최적화를 한 후]
        inputs = torch.from_numpy(inputs).float()
        outputs = net(inputs)
        print("output : {outputs}, labels : {labels}" )
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 통계를 출력합니다.
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')

PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

dataiter = iter(test_data)
images, labels = dataiter.next()

# 이미지를 출력합니다.
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join(f'{label_names[labels[j]]:5s}' for j in range(4)))