from torchvision import datasets, transforms

# reference:https://github.com/weiaicunzai/pytorch-cifar100/blob/master/utils.py            
def create_dataset(data_path):
    normalize = transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), 
                                                    (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
    train_transform = transforms.Compose([
                                transforms.RandomCrop(32, 4),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomRotation(15),
                                transforms.ToTensor(),
                                normalize
                            ])
    test_transform = transforms.Compose([
                            transforms.ToTensor(),
                            normalize
                            ])
    train_dataset = datasets.CIFAR100(data_path, train = True, 
                                            download = True, transform=train_transform)
    test_dataset = datasets.CIFAR100(data_path, train = False, 
                                            download = True, transform=test_transform)
    return train_dataset, test_dataset