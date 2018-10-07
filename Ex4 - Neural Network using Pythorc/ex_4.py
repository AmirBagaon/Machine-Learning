#204313100
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms




# Regular net with no additions
class RegularNet(nn.Module):
    def __init__(self, image_size):
        super(RegularNet, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, HL1size)
        self.fc1 = nn.Linear(HL1size, HL2size)
        self.fc2 = nn.Linear(HL2size, 10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

#Net - with batch normal
class BatchNormalNet(nn.Module):
    def __init__(self, image_size):
        super(BatchNormalNet, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, HL1size)
        self.bn1 = nn.BatchNorm1d(HL1size)
        self.fc1 = nn.Linear(HL1size, HL2size)
        self.bn2 = nn.BatchNorm1d(HL2size)
        self.fc2 = nn.Linear(HL2size, 10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.bn1(self.fc0(x)))
        x = F.relu(self.bn2(self.fc1(x)))
        x = self.fc2(x)
        return F.log_softmax(x, dim = 1)



#Net - with dropout
class DropoutNet(nn.Module):
    def __init__(self,image_size):
        super(DropoutNet, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, HL1size)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(HL1size, HL2size)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(HL2size, 10) #10 - Total classes
    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.dropout(x) #Added dropout
        x = F.relu(self.fc1(x))
        x = F.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim = 1)




def train(model):
    """
    train the model
    :param model: model
    """
    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, labels)
        loss.backward()
        optimizer.step()

def test(specify_loader):
    """
    Activate test on the input (some loader)
    Calculate loss and accuracy
    :param specify_loader: loader
    :return: loss and accuracy
    """
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in specify_loader:
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average = False).data[0]
        pred = output.data.max(1, keepdim= True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    return test_loss, correct   #loss Will be devided after

def finalTest(specify_loader):
    """
    Activate test on test loader and write to test.pred
    :param specify_loader: test
    """
    model.eval()
    test_loss = 0
    correct = 0
    pred_file = open("test.pred", 'w')
    for data, target in specify_loader:
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0]
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        pred_file.write(str(pred.item()) + "\n")

    test_loss /= (len(specify_loader))
    print('\n Test Set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, (len(specify_loader)),
        100. * correct / (len(specify_loader))))

    pred_file.close()

###MAIN

#HyperParams

HL1size = 100
HL2size = 50
lrnRate = 0.007
batch_size = 32
epochs = 10

###Several info was taken from: https://am207.github.io/2018spring/wiki/ValidationSplits.html
transforms = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])

## Define our FashionMNIST Datasets (Images and Labels) for training and testing
train_dataset = datasets.FashionMNIST(root='./data', train=True, transform=transforms, download=True)
test_dataset = datasets.FashionMNIST(root='./data', train=False, transform=transforms)

# Define the indices and split our training dataset into training and validation sets
indices = list(range(len(train_dataset))) # start with all the indices in training set
ratio = (len(train_dataset) * 0.2)
split = int(ratio)

# Random, non-contiguous split
validation_idx = np.random.choice(indices, size=split, replace=False)
train_idx = list(set(indices) - set(validation_idx))

train_sampler = SubsetRandomSampler(train_idx)
validation_sampler = SubsetRandomSampler(validation_idx)

# Create the train_loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1, sampler=validation_sampler)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

#Choose Model and optimizer
model = BatchNormalNet(image_size=28*28)
optimizer = optim.SGD(model.parameters(), lr=lrnRate)

trainLossAvg = []
trainAcc = []
validLossAvg = []
validAcc = []
print("OK")
#start train
for epoch in range(epochs):
    print("Epoch: " , epoch + 1)
    train(model)
    train_loss, train_correct = test(train_loader)
    trainLossAvg.append(train_loss / len(train_idx))
    trainAcc.append(100. * train_correct / (len(train_loader) * batch_size))
    print('\n' + "train set" + ': Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        train_loss, train_correct, (len(train_loader) * batch_size),
        100. * train_correct / (len(train_loader) * batch_size)))

    valid_loss, valid_correct = test(validation_loader)
    validLossAvg.append(valid_loss / len(validation_loader))
    validAcc.append(100. * valid_correct / len(validation_loader))
    print('\n' + "validation set: " + ': Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        valid_loss, valid_correct, len(validation_loader),
        100. * valid_correct / len(validation_loader)))

#Test the test loader
finalTest(test_loader)

#Create avgloss graph
plt.title("Model with batch normalize - Avg Loss Per Epoch")
line1, = plt.plot(range(1,11), trainLossAvg, "green", label='train')
line2, = plt.plot(range(1,11), validLossAvg, "orange", label='validation')
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Avg loss")
plt.show()

#Create acc graph
plt.title("Model with batch normalize - Accuracy Rate (out of 100%) Per Epoch")
line1, = plt.plot(range(1,11), trainAcc, "green", label='trainAccuracy')
line2, = plt.plot(range(1,11), validAcc, "orange", label='validationAccuracy')
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.show()


