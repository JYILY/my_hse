import argparse
import logging
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed
from torchvision import datasets, transforms
import torchvision
from torchtoolbox.tools import mixup_data, mixup_criterion
from models.ResNetEmbed import ResNetEmbed
from tqdm import tqdm
from utils import dataset

data_root = {
    "train": "./data/mini_cub/train/",
    "test": "./data/mini_cub/val/"
}

data_transform = {
    "train": transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # transforms.GaussianBlur(),
        # transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    "test": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    )}

save_path = "./trains/" + "[" + time.strftime("%Y-%m-%d %H-%M-%S", time.localtime()) + "]"
os.makedirs(save_path)
save_path += "/"

logger = logging.getLogger(__name__)
handler = logging.FileHandler(save_path + "log.txt")
console = logging.StreamHandler()

logger.addHandler(handler)
logger.addHandler(console)

levelTupleList = [("order", 13), ("family", 37), ("genus", 122), ("class", 200)]


def get_time(sec):
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)


def train(opt):
    logger.info(opt)

    num_level = len(levelTupleList)

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.device
    lr, batch_size, epochs, train_level = opt.lr, opt.batch_size, opt.epochs, opt.train_level

    train_loader, num_train = dataset.create_dataloader(root=data_root['train'],
                                                        batch_size=batch_size,
                                                        transform=data_transform['train'])
    test_loader, num_test = dataset.create_dataloader(root=data_root['test'],
                                                      batch_size=batch_size,
                                                      transform=data_transform['test'])
    num_data = [num_train, num_test]

    train_bar = tqdm(train_loader)
    test_bar = tqdm(test_loader)

    net = ResNetEmbed(levelTupleList, pretrained=not opt.nopretrain)
    if opt.adam:
        optimizer = optim.Adam(net.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.00005)
    loss_function = nn.CrossEntropyLoss()
    hierarchyUtils = dataset.HierarchyUtil()

    train_acc_list = [[] for i in range(num_level)]
    val_acc_list = [[] for i in range(num_level)]

    train_loss_list = [[] for i in range(num_level)]
    val_loss_list = [[] for i in range(num_level)]

    for epoch in range(1, epochs + 1):

        net.train()
        train_bar.desc = f"train epoch[{epoch}/{epochs}]"
        for featrues, labels in train_bar:
            featrues = featrues.cuda()
            labels = labels.cuda()
            optimizer.zero_grad()
            results = net(featrues)
            labels = hierarchyUtils.get_hierarchy(labels)
            loss = loss_function(results[train_level], labels[train_level])
            loss.backward()
            optimizer.step()

        net.val()
        total_loss = [0.0, 0.0]
        acc = [0.0, 0.0]
        with torch.no_grad():
            for idx, loader in enumerate([train_loader, test_loader]):
                for featrues, labels in loader:
                    featrues = featrues.cuda()
                    labels = labels.cuda()
                    results = net(featrues)[train_level]
                    labels = hierarchyUtils.get_hierarchy(labels)[train_level]
                    val_loss = loss_function(results, labels)
                    total_loss[idx] += val_loss.item() / num_data[idx]
                    y_hat = torch.max(results, dim=1)[1]
                    acc[idx] += torch.eq(y_hat, labels).sum().item()
            print(f"train Loss: {round(total_loss[0], 5)}  train Acc: {round(acc[0], 5)}")
            print(f"test  Loss: {round(total_loss[1], 5)}  test  Acc: {round(acc[0], 5)}")


    # loader = dataset.create_dataloader(root=data_root['train'], batch_size=1, transform=data_transform['train'])
    # net = ResNetEmbed(levelTupleList, pretrained=True)
    # loss_function = nn.CrossEntropyLoss()
    # hierarchyUtils = dataset.HierarchyUtil()
    # optimizer = torch.optim.SGD(net.parameters(),lr=0.01)
    #
    # train_bar = tqdm(loader)
    # for X, y in train_bar:
    #     # cls_hat, genus_hat, family_hat, order_hat = net(X)
    #     loss = None
    #     for hierarchy ,hat in zip(hierarchyUtils.get_hierarchy(y),net(X)):
    #         if loss is None:
    #             loss = loss_function(hat, hierarchy)
    #         else :
    #             loss  = loss + loss_function(hat, hierarchy)
    #     loss.backward()
    #     optimizer.step()
    #     train_bar.desc = f"loss:{round(loss.item(), 4)}"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--nopretrain', action='store_true', help='resnet50 pretrained is false')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--train-level', type=int, default=0, help='training level')

    train(parser.parse_args())

    # net = ResNetEmbed(levelTupleList, pretrained=not opt.nopretrain)
    # if opt.weights != '':
    #     net.load_state_dict(torch.load(opt.weights))
    # # torch.distributed.init_process_group(backend='nccl', rank=0, world_size=1,init_method='tcp://localhost:6868')
    # # net = nn.parallel.DistributedDataParallel(net.cuda())
    # net = torch.nn.DataParallel(net.cuda())
    #
    # optimizer = optim.SGD(net.parameters(),lr=lr,momentum=0.9,weight_decay=0.00005)
    # loss_function = nn.CrossEntropyLoss()
