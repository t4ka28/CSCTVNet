from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from os import listdir, path
from PIL import Image
import torch
import IPython

class MyDataset(Dataset):
    def __init__(self, root_dirs, transform=None, verbose=False):
        """
        Args:
            root_dirs (string): A list of directories with all the images' folders.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dirs = root_dirs
        self.transform = transform
        self.images_path = []
        for cur_path in root_dirs:
            self.images_path += [path.join(cur_path, file) for file in listdir(cur_path) if file.endswith(('png','jpg','jpeg','bmp'))]
        self.verbose = verbose

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        img_name = self.images_path[idx]
        image = Image.open(img_name)
        if image.mode == "RGBA":
            image = image.convert("RGB")
        
        if self.transform:
            image = self.transform(image)

        if self.verbose:
            return image, img_name.split('/')[-1]

        return image


def get_dataloaders(path_list, n_ch=3, crop_size=128, batch_size=1, phase="train"):
    generator = torch.Generator(device='cuda:0')
    batch_sizes = {'train': batch_size, 'valid':1, 'test':1}
    
    if n_ch == 1:
        train_transforms = transforms.Compose([
            transforms.RandomCrop(crop_size),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()])

        valid_transforms = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()])

        test_transforms = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()])
    else:
        train_transforms = transforms.Compose([
            transforms.RandomCrop(crop_size),
            transforms.ToTensor()])

        valid_transforms = transforms.Compose([
            transforms.ToTensor()])

        test_transforms = transforms.Compose([
            transforms.ToTensor()])

    if phase == "train":
        train_path_list, valid_path_list, test_path_list = path_list
        data_transforms = {'train': train_transforms, 'valid': valid_transforms, 'test': test_transforms}
        image_datasets = {'train': MyDataset(train_path_list, data_transforms['train']), 'valid': MyDataset(valid_path_list, data_transforms['valid']), 'test': MyDataset(test_path_list, data_transforms['test'])}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_sizes[x], shuffle=(phase=='train'), generator=generator) for x in ['train', 'valid', 'test']}
        
    elif phase == "test":
        test_path_list = path_list[0]
        data_transforms = test_transforms
        image_datasets = MyDataset(test_path_list, data_transforms)
        dataloaders = {"test": torch.utils.data.DataLoader(image_datasets, batch_size=1, shuffle=0)}
        
    return dataloaders