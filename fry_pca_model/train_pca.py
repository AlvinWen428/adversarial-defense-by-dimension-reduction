import __init__paths
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm
import os

from tensorboardX import SummaryWriter
from model.vgg_tiny import Conv, Fc
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA


BATCH_SIZE = 50
LEARNING_RATE = 0.01
PRETRAIN_EPOCH = 20
EPOCH = 50
with_isomap = True
isomap_number = 300
n_components = 30
n_neighbors = 5

write_log = True

train_transform = transforms.Compose([
    transforms.RandomCrop(28, padding=4),
    transforms.RandomRotation(20),
    transforms.ToTensor()])
test_transform = transforms.Compose([transforms.ToTensor()])


def train(epoch, model, data_loader, optim, criterion, cfg, writer, file_name):
    assert cfg == 'pre_train' or cfg == 'isomap'
    model['conv'].train()
    model['pretrain_fc'].train()
    model['isomap_fc'].train()

    features_number = 0
    correct = 0
    total = 0
    total_loss = 0
    isomap_flag = False
    for batch_idx, (data, target) in enumerate(tqdm(data_loader)):
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        optim.zero_grad()

        # judge the train phase
        if cfg == 'pre_train':
            features = model['conv'](data)
            output = model['pretrain_fc'](features)

            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            total += target.shape[0]

            loss = criterion(output, target)
            total_loss += loss.item()
            loss.backward()
            optim.step()
        else:
            features = model['conv'](data)
            if features_number == 0:
                features_tensor = features.cpu().detach()
                target_tensor = target
                features_number += BATCH_SIZE
            elif features_number < isomap_number - BATCH_SIZE:
                features_tensor = torch.cat((features_tensor, features.cpu().detach()), dim=0)
                target_tensor = torch.cat((target_tensor, target), dim=0)
                features_number += BATCH_SIZE
            else:
                if not isomap_flag:
                    features_array = torch.cat((features_tensor, features.cpu().detach()), dim=0).numpy()
                    target_tensor = torch.cat((target_tensor, target), dim=0)
                    if torch.cuda.is_available():
                        transformed = torch.Tensor(model['isomap'].fit_transform(features_array)).cuda()
                    else:
                        transformed = torch.Tensor(model['isomap'].fit_transform(features_array))
                    output = model['isomap_fc'](transformed)
                else:
                    features_array = torch.cat((features_tensor[:isomap_number-BATCH_SIZE], features.cpu().detach()), dim=0).numpy()
                    target_tensor = target
                    if torch.cuda.is_available():
                        transformed = torch.Tensor(model['isomap'].fit_transform(features_array)[isomap_number-BATCH_SIZE:]).cuda()
                    else:
                        transformed = torch.Tensor(model['isomap'].fit_transform(features_array)[isomap_number-BATCH_SIZE:])
                    output = model['isomap_fc'](transformed)

                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target_tensor.data.view_as(pred)).cpu().sum()
                total += target_tensor.shape[0]

                loss = criterion(output, target_tensor)
                total_loss += loss.item()
                loss.backward()
                optim.step()
                isomap_flag = True

        if batch_idx % (isomap_number // BATCH_SIZE) == isomap_number // BATCH_SIZE - 1:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Acc:[{}/{}] {:.6f} Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(data_loader.dataset),
                100. * batch_idx / len(data_loader), int(correct), int(total), float(correct)/float(total)*100,
                loss.item()))
    log = ('Train Epoch: {}\tAcc:[{}/{}]{:.6f} Loss: {:.6f}'.format(
                epoch, int(correct), int(total), float(correct)/float(total)*100, total_loss))
    print(log)
    if write_log:
        if cfg == 'pre_train':
            writer.add_scalar('Pretrain_train_loss', total_loss, epoch)
            writer.add_scalar('Pretrain_train_accuracy', float(correct) / float(total) * 100, epoch)
            torch.save(model['conv'].state_dict(),
                       './model_saved/{}/conv_{}_epoch_{}.pkl'.format(file_name, file_name, epoch))
            torch.save(model['pretrain_fc'].state_dict(),
                       './model_saved/{}/pretrainfc_{}_epoch_{}.pkl'.format(file_name, file_name, epoch))

        else:
            writer.add_scalar('Isomap_train_loss', total_loss, epoch)
            writer.add_scalar('Isomap_train_accuracy', float(correct) / float(total) * 100, epoch)
            torch.save(model['isomap_fc'].state_dict(),
                       './model_saved/{}/pretrainfc_{}_epoch_{}.pkl'.format(file_name, file_name, epoch))


def test(epoch, model, train_loader, test_loader, criterion, cfg, writer, file_name):
    assert cfg == 'pre_train' or cfg == 'isomap'
    model['conv'].eval()
    model['pretrain_fc'].eval()
    model['isomap_fc'].eval()

    features_number = 0
    correct = 0
    total = 0
    total_loss = 0
    isomap_flag = False
    for batch_idx, (data, target) in enumerate(tqdm(test_loader)):
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()

        # judge the train phase
        if cfg == 'pre_train':
            features = model['conv'](data)
            output = model['pretrain_fc'](features)

            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            total += target.shape[0]

            loss = criterion(output, target)
            total_loss += loss.item()
        else:
            features = model['conv'](data)
            if features_number == 0:
                features_tensor = features.cpu().detach()
                target_tensor = target
                features_number += BATCH_SIZE
            elif features_number < isomap_number - BATCH_SIZE:
                features_tensor = torch.cat((features_tensor, features.cpu().detach()), dim=0)
                target_tensor = torch.cat((target_tensor, target), dim=0)
                features_number += BATCH_SIZE
            else:
                if not isomap_flag:
                    features_array = torch.cat((features_tensor, features.cpu().detach()), dim=0).numpy()
                    target_tensor = torch.cat((target_tensor, target), dim=0)
                    if torch.cuda.is_available():
                        transformed = torch.Tensor(model['isomap'].fit_transform(features_array)).cuda()
                    else:
                        transformed = torch.Tensor(model['isomap'].fit_transform(features_array))
                    output = model['isomap_fc'](transformed)
                else:
                    features_array = torch.cat((features_tensor[:isomap_number - BATCH_SIZE], features.cpu().detach()),
                                               dim=0).numpy()
                    target_tensor = target
                    if torch.cuda.is_available():
                        transformed = torch.Tensor(model['isomap'].fit_transform(features_array)[isomap_number - BATCH_SIZE:]).cuda()
                    else:
                        transformed = torch.Tensor(model['isomap'].fit_transform(features_array)[isomap_number - BATCH_SIZE:])
                    output = model['isomap_fc'](transformed)

                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target_tensor.data.view_as(pred)).cpu().sum()
                total += target_tensor.shape[0]

                loss = criterion(output, target_tensor)
                total_loss += loss.item()

    log = ('Test Epoch: {}\tAcc:[{}/{}]{:.6f} Loss: {:.6f}'.format(
                epoch, int(correct), int(total), float(correct)/float(total)*100, total_loss))
    print(log)
    if write_log:
        if cfg == 'pre_train':
            writer.add_scalar('Pretrain_test_loss', total_loss, epoch)
            writer.add_scalar('Pretrain_test_accuracy', float(correct) / float(total) * 100, epoch)
        else:
            writer.add_scalar('Isomap_test_loss', total_loss, epoch)
            writer.add_scalar('Isomap_test_accuracy', float(correct) / float(total) * 100, epoch)


def main():
    file_name = 'train_isomap_pureconv_stableisomap_downdimension_{}_isomap_number_{}_neighbors_{}_batch_{}'.format(
        n_components, isomap_number, n_neighbors, BATCH_SIZE)
    if write_log:
        writer = SummaryWriter('./runs/{}'.format(file_name))
        if not os.path.exists('./model_saved/{}'.format(file_name)):
            os.makedirs('./model_saved/{}'.format(file_name))
    else:
        writer = None

    data_train = datasets.MNIST(root="./data/", transform=train_transform, train=True, download=True)
    data_test = datasets.MNIST(root="./data/", transform=test_transform, train=False)

    train_loader = torch.utils.data.DataLoader(dataset=data_train, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=data_test, batch_size=BATCH_SIZE, shuffle=False)

    conv_feature = Conv()
    pretrain_fc = Fc(input_channel=1568, output_channel=10)
    # isomap = Isomap(n_components=n_components, n_neighbors=n_neighbors)
    isomap = PCA(n_components=n_components)
    isomap_fc = Fc(input_channel=n_components, output_channel=10)

    model = {'conv': conv_feature, 'pretrain_fc': pretrain_fc, 'isomap': isomap, 'isomap_fc': isomap_fc}
    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model['conv'], model['pretrain_fc'], model['isomap_fc'] = model['conv'].cuda(), model['pretrain_fc'].cuda(), \
                                                                  model['isomap_fc'].cuda()
        criterion = criterion.cuda()

    #model['conv'].load_state_dict(torch.load('./model_saved/conv_train_isomap_pureconv_SGD_downdimension_50_isomap_number_1000_neighbors_7_epoch_19.pkl'))
    pretrain_optim = optim.SGD(list(conv_feature.parameters()) + list(pretrain_fc.parameters()), lr=0.1, momentum=0.9)
    isomap_optim = optim.SGD(isomap_fc.parameters(), lr=0.1, momentum=0.9)

    for epoch in range(PRETRAIN_EPOCH):
        cfg = 'pre_train'
        train(epoch, model, train_loader, pretrain_optim, criterion, cfg, writer, file_name)
        test(epoch, model, train_loader, test_loader, criterion, cfg, writer, file_name)
    for epoch in range(EPOCH):
        cfg = 'isomap'
        train(epoch, model, train_loader, isomap_optim, criterion, cfg, writer, file_name)
        test(epoch, model, train_loader, test_loader, criterion, cfg, writer, file_name)

    if write_log:
        writer.close()


if __name__ == '__main__':
    main()