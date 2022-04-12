import os

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from image_folder import image_folder_num


def data_loader(class_list_num, root, batch_size, workers):
    traindir = os.path.join(root, 'train')
    valdir = os.path.join(root, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    
    train_data = image_folder_num(
        traindir,
        class_list_num,
        transform = transform,
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=batch_size, 
        shuffle=True,
        num_workers=workers, 
        pin_memory=True,
        sampler=None
    )

    val_data = image_folder_num(
        valdir,
        class_list_num,
        transform = transform,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_data, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=workers, 
        pin_memory=True
    )

    # train_dataset = datasets.ImageFolder(
    #     traindir,
    #     transforms.Compose([
    #         transforms.RandomResizedCrop(224),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         normalize
    #     ])
    # )
    # val_dataset = datasets.ImageFolder(
    #     valdir,
    #     transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         normalize
    #     ])
    # )

    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     batch_size=batch_size,
    #     shuffle=True,
    #     num_workers=workers,
    #     pin_memory=pin_memory,
    #     sampler=None
    # )
    # val_loader = torch.utils.data.DataLoader(
    #     val_dataset,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     num_workers=workers,
    #     pin_memory=pin_memory
    # )
    return train_loader, val_loader

class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            self.next_input = self.next_input.float()
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)
            
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target