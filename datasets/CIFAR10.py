from torchvision import datasets, transforms

def create_dataset(data_path):
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])
    train_transform = transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomCrop(32, 4),
                        transforms.ToTensor(),
                        normalize
                        ])
    test_transform = transforms.Compose([
                    transforms.ToTensor(),
                    normalize
                    ])
    train_dataset = datasets.CIFAR10(data_path, train = True, 
                                            download = True, transform=train_transform)
    test_dataset = datasets.CIFAR10(data_path, train = False, 
                                            download = True, transform=test_transform)
    return train_dataset, test_dataset