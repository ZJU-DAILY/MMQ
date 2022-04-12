import os
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from prefetch_generator import BackgroundGenerator

from image_folder import *

class DataLoaderX(torch.utils.data.DataLoader):
    
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def data_loader(class_list_num, root, batch_size, workers, rate):
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
    
    train_data = image_folder_num_train(
        traindir,
        class_list_num,
        transform = transform,
        rate = rate
    )
    train_loader = torch.utils.data.DataLoader(
    # train_loader = DataLoaderX(
        dataset=train_data,  
        batch_size=batch_size,  
        # shuffle=True,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        sampler=None
    )
    
    val_data = image_folder_num_val(
        traindir,
        class_list_num,
        transform = transform,
        rate = rate
    )
    val_loader = torch.utils.data.DataLoader(
    # val_loader = DataLoaderX(
        dataset=val_data, 
        batch_size=batch_size,  
        shuffle=True,
        num_workers=workers, 
        pin_memory=True,
        sampler=None
    )

    test_data = image_folder_num(
        valdir,
        class_list_num,
        transform = transform,
    )
    test_loader = torch.utils.data.DataLoader(
    # test_loader = DataLoaderX(
        dataset=test_data, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=workers,
        pin_memory=True
    )
    return train_loader, val_loader, test_loader

def data_loader_o(class_list_num, root, batch_size, workers):
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
    return train_loader, val_loader



class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

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