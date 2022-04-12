from torchvision.datasets.folder import *
import random


def subset(alist, idxs):
    sub_list = []
    for idx in idxs:
        sub_list.append(alist[idx])

    return sub_list

class image_folder_num_train(DatasetFolder):
    def __init__(self, root, class_list_num, transform=None, target_transform=None,
                 loader=default_loader, rate = 1.0):
        if rate >= 0.99:
            print("image_folder_num_train error")
            exit(0)

        classes, class_to_idx = self.find_classes(root, class_list_num)
        imgs = make_dataset(root, class_to_idx, IMG_EXTENSIONS)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        r=random.random
        random.seed(1)
        random.shuffle(imgs,random=r)

        l = round(rate*len(imgs))
        imgs = imgs[:l]

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
    
    def find_classes(self, dir, class_list_num):
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        classes = subset(classes, class_list_num)
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        index (int): Index
	Returns:tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.imgs)

class image_folder_num_val(DatasetFolder):
    def __init__(self, root, class_list_num, transform=None, target_transform=None,
                 loader=default_loader, rate = 0.0):
        if rate <= 0.01:
            print("image_folder_num_val error")
            exit(0)

        classes, class_to_idx = self.find_classes(root, class_list_num)
        imgs = make_dataset(root, class_to_idx, IMG_EXTENSIONS)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        r=random.random
        random.seed(1)
        random.shuffle(imgs,random=r)
        
        l = round(rate*len(imgs))
        imgs = imgs[l:]

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
    
    def find_classes(self, dir, class_list_num):
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        classes = subset(classes, class_list_num)
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        index (int): Index
	Returns:tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.imgs)

class image_folder_num(DatasetFolder):
    def __init__(self, root, class_list_num, transform=None, target_transform=None,
                 loader=default_loader):
        classes, class_to_idx = self.find_classes(root, class_list_num)
        imgs = make_dataset(root, class_to_idx, IMG_EXTENSIONS)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        r=random.random
        random.seed(1)
        random.shuffle(imgs,random=r)

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
    
    def find_classes(self, dir, class_list_num):
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        classes = subset(classes, class_list_num)
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        index (int): Index
	Returns:tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.imgs)