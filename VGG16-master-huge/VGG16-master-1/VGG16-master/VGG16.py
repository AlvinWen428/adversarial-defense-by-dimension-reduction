import torch
import torch.nn as tnn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

BATCH_SIZE = 32
LEARNING_RATE = 0.01
EPOCH = 100

transform = transforms.Compose([
    transforms.RandomResizedCrop(28),
    transforms.ToTensor()])

data_train = dsets.MNIST(root = "./data/",
                         transform=transform,
                            train = True,
                            download = True)

data_test = dsets.MNIST(root="./data/",
                        transform=transform,
                           train = False)

trainLoader = torch.utils.data.DataLoader(dataset=data_train, batch_size=BATCH_SIZE, shuffle=True)
testLoader = torch.utils.data.DataLoader(dataset=data_test, batch_size=BATCH_SIZE, shuffle=False)

class VGG16(tnn.Module):
  def __init__(self):
    super(VGG16, self).__init__()
    self.layer1 = tnn.Sequential(

        # 1-1 conv layer
        tnn.Conv2d(1, 64, kernel_size=3, padding=1),
        tnn.BatchNorm2d(64),
        tnn.ReLU(),

        # 1-2 conv layer
        tnn.Conv2d(64, 64, kernel_size=3, padding=1),
        tnn.BatchNorm2d(64),
        tnn.ReLU(),

        # 1 Pooling layer
        tnn.MaxPool2d(kernel_size=2, stride=2))

    self.layer2 = tnn.Sequential(

        # 2-1 conv layer
        tnn.Conv2d(64, 128, kernel_size=3, padding=1),
        tnn.BatchNorm2d(128),
        tnn.ReLU(),

        # 2-2 conv layer
        tnn.Conv2d(128, 128, kernel_size=3, padding=1),
        tnn.BatchNorm2d(128),
        tnn.ReLU(),

        # 2 Pooling lyaer
        tnn.MaxPool2d(kernel_size=2, stride=2))

    self.layer3 = tnn.Sequential(

        # 3-1 conv layer
        tnn.Conv2d(128, 256, kernel_size=3, padding=1),
        tnn.BatchNorm2d(256),
        tnn.ReLU(),

        # 3-2 conv layer
        tnn.Conv2d(256, 256, kernel_size=3, padding=1),
        tnn.BatchNorm2d(256),
        tnn.ReLU(),

        # 3 Pooling layer
        tnn.MaxPool2d(kernel_size=2, stride=2))

    self.layer4 = tnn.Sequential(

        # 4-1 conv layer
        tnn.Conv2d(256, 512, kernel_size=3, padding=1),
        tnn.BatchNorm2d(512),
        tnn.ReLU(),

        # 4-2 conv layer
        tnn.Conv2d(512, 512, kernel_size=3, padding=1),
        tnn.BatchNorm2d(512),
        tnn.ReLU(),

        # 4 Pooling layer
        tnn.MaxPool2d(kernel_size=2, stride=2))

    # self.layer5 = tnn.Sequential(
    #
    #     # 5-1 conv layer
    #     tnn.Conv2d(512, 512, kernel_size=3, padding=1),
    #     tnn.BatchNorm2d(512),
    #     tnn.ReLU(),
    #
    #     # 5-2 conv layer
    #     tnn.Conv2d(512, 512, kernel_size=3, padding=1),
    #     tnn.BatchNorm2d(512),
    #     tnn.ReLU(),
    #
    #     # 5 Pooling layer
    #    tnn.MaxPool2d(kernel_size=2, stride=2))

    self.layer6 = tnn.Sequential(

        # 6 Fully connected layer
        # Dropout layer omitted since batch normalization is used.
        tnn.Linear(512, 512),
        tnn.BatchNorm1d(512),
        tnn.ReLU())
        

    self.layer7 = tnn.Sequential(

        # 7 Fully connected layer
        # Dropout layer omitted since batch normalization is used.
        tnn.Linear(512, 512,
        tnn.BatchNorm1d(512)),
        tnn.ReLU())

    self.layer8 = tnn.Sequential(

        # 8 output layer
        tnn.Linear(512, 10),
        tnn.BatchNorm1d(10),
        tnn.Softmax())

  def forward(self, x):
      out = self.layer1(x)
      out = self.layer2(out)
      out = self.layer3(out)
      out = self.layer4(out)
   #   out = self.layer5(out)
      vgg16_features = out.view(out.size(0), -1)
      out = self.layer6(vgg16_features)
      out = self.layer7(out)
      out = self.layer8(out)

      return out

      
vgg16 = VGG16()
vgg16.cuda()

# Loss and Optimizer
cost = tnn.CrossEntropyLoss()
optimizer = torch.optim.Adam(vgg16.parameters(), lr=LEARNING_RATE)

# Train the model
for epoch in range(EPOCH):
#  for i, (images, labels) in enumerate(trainLoader):
  vgg16.train()
  correct = 0
  total = 0
  for images, labels in trainLoader:
    images = Variable(images).cuda()
    labels = Variable(labels).cuda()

    # Forward + Backward + Optimize
    optimizer.zero_grad()
    outputs = vgg16(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted.cpu() == labels.cpu()).sum()
    loss = cost(outputs, labels.cuda())
    loss.backward()
    optimizer.step()

  print ('Epoch [%d/%d], Loss. %.4f' %
             (epoch+1, EPOCH, loss.data[0]))
  print('Test Accuracy of the model on the training set: %d %%' % (100 * correct / total))

# Test the model
  vgg16.eval()
  correct = 0
  total = 0

  for images, labels in testLoader:
    images = Variable(images).cuda()
    outputs = vgg16(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted.cpu() == labels).sum()

  print('Test Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total))

# Save the Trained Model
torch.save(vgg16.state_dict(),'checkpoint_without_model.pt')
