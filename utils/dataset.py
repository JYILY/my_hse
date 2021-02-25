import torch
import torchvision
from torchvision import datasets, transforms
import os


def create_dataloader(root, batch_size, transform, workers=0, world_size=1, shuffle=True, pin_memory=False):
    dataset = torchvision.datasets.ImageFolder(root, transform=transform)
    nw = min([os.cpu_count() // world_size, batch_size if batch_size > 1 else 0, workers])
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=batch_size,
                                         shuffle=shuffle,
                                         pin_memory=pin_memory,
                                         num_workers=nw)

    return loader,len(dataset)


class HierarchyUtil:
    def __init__(self, path="./data/cub_200_2011/level.txt"):
        self.levels = []
        with open(path, "r") as f:
            for line in f.readlines():
                cls, genus, family, order = [int(s) for s in line.strip().split(' ')]
                self.levels.append([cls, genus, family, order])

    def get_hierarchy(self,cls):
        hierarchy = [[] for i in range(4)]
        device = 4
        for c in cls:
            for idx, x in enumerate(self.levels[c.item()]):
                hierarchy[idx].append(x)

        return [torch.tensor(x).to(cls.device()) for x in hierarchy]


