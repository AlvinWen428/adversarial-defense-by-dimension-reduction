import torch
import torch.nn as nn
from sklearn.decomposition import PCA
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm


class Conv(nn.Module):
    def __init__(self):
        super(Conv, self).__init__()
        self.layer1 = nn.Sequential(

            # 1-1 conv layer
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            # 1-2 conv layer
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),

            # 1 Pooling layer
            nn.AvgPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(

            # 1-1 conv layer
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            # 1-2 conv layer
            nn.Conv2d(32, 8, kernel_size=3, padding=1))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        features = out.view(out.shape[0], -1)
        return features


class Fc(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(Fc, self).__init__()
        self.layer = nn.Linear(input_channel, output_channel)
        self.layer = nn.Sequential(
            nn.Linear(input_channel, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),

            nn.Linear(128, output_channel))

    def forward(self, x):
        out = self.layer(x)
        return out


class ConvNet(nn.Module):
    def __init__(self, pre_trained=True):
        super(ConvNet, self).__init__()
        self.conv = Conv()
        self.fc = Fc(input_channel=1568, output_channel=10)
        if pre_trained:
            self.conv.load_state_dict(torch.load('./conv_net.pkl'))
            self.fc.load_state_dict(torch.load('./baseline_fc.pkl'))

    def forward(self, x):
        feature = self.conv(x)
        output = self.fc(feature)

        return output


class ConvPCANet(nn.Module):
    def __init__(self, n_components=10, pca_number=2000, batch_size=50, use_cuda=True):
        super(ConvPCANet, self).__init__()
        if use_cuda:
            self.conv = Conv().cuda()
            self.fc = Fc(input_channel=n_components, output_channel=10).cuda()
        else:
            self.conv = Conv()
            self.fc = Fc(input_channel=n_components, output_channel=10)

        self.pca = PCA(n_components=n_components, copy=True)

        self.conv.load_state_dict(torch.load('/DATA4_DB3/data/wen/isomap/workspace/one-pixel-attack-pytorch/models/conv_net.pkl'))
        self.fc.load_state_dict(torch.load('/DATA4_DB3/data/wen/isomap/workspace/one-pixel-attack-pytorch/models/pca_fc.pkl'))

        features_number = 0
        self.pca_number = pca_number
        self.batch_size = batch_size
        self.use_cuda = use_cuda
        train_transform = transforms.Compose([transforms.ToTensor()])
        data_train = datasets.MNIST(root="./data/", transform=train_transform, train=True, download=True)
        train_loader = torch.utils.data.DataLoader(dataset=data_train, batch_size=batch_size, shuffle=False)
        for batch_idx, (data, target) in enumerate(train_loader):
            if use_cuda:
                data, target = data.cuda(), target.cuda()

            features = self.conv(data)
            if features_number == 0:
                self.features_tensor = features.cpu().detach()
                features_number += batch_size
            elif features_number < pca_number - batch_size:
                self.features_tensor = torch.cat((self.features_tensor, features.cpu().detach()), dim=0)
                features_number += batch_size
            else:
                features_array = self.features_tensor.numpy()
                self.pca.fit(features_array)
                break

    def __call__(self, x):
        features = self.conv(x)
        features_array = torch.cat((self.features_tensor[:self.pca_number - self.batch_size], features.cpu().detach()),
                                   dim=0).numpy()
        if self.use_cuda:
            transformed = torch.Tensor(self.pca.transform(features_array)[self.pca_number - self.batch_size:]).cuda()
        else:
            transformed = torch.Tensor(self.pca.transform(features_array)[self.pca_number - self.batch_size:])
        output = self.fc(transformed)
        return output


if __name__ == '__main__':
    BATCH_SIZE = 50
    model1 = Conv().cuda()
    model2 = Fc(input_channel=1568, output_channel=10).cuda()
    model3 = ConvNet(pre_trained=True).cuda()
    model4 = ConvPCANet().cuda()
    test_input = torch.Tensor(50, 1, 28, 28).cuda()
    output1 = model1(test_input)
    print(output1.shape)
    output2 = model2(output1)
    print(output2.shape)
    output3 = model3(test_input)
    print(output3.shape)
    output4 = model4(test_input)
    print(output4.shape)

    train_transform = transforms.Compose([transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToTensor()])

    data_train = datasets.MNIST(root="./data/", transform=train_transform, train=True, download=True)
    data_test = datasets.MNIST(root="./data/", transform=test_transform, train=False)

    train_loader = torch.utils.data.DataLoader(dataset=data_train, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=data_test, batch_size=BATCH_SIZE, shuffle=False)

    baseline_correct = 0
    pca_correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(tqdm(test_loader)):
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()

        # judge the train phase
        baseline_output = model3(data)

        baseline_pred = baseline_output.data.max(1, keepdim=True)[1]
        baseline_correct += baseline_pred.eq(target.data.view_as(baseline_pred)).cpu().sum()
        total += target.shape[0]

        pca_output = model4(data)
        pca_pred = pca_output.data.max(1, keepdim=True)[1]
        pca_correct += pca_pred.eq(target.data.view_as(pca_pred)).cpu().sum()

    print('baseline: {:.6f}%'.format(float(baseline_correct) / len(test_loader.dataset) * 100))
    print('pca: {:.6f}%'.format(float(pca_correct) / len(test_loader.dataset) * 100))