from torchvision import datasets, transforms

def create_dataset(data_path):
    dataset_transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                         ])
    train_dataset = datasets.FashionMNIST(data_path, train = True, 
                                            download = True, transform=dataset_transform)
    test_dataset = datasets.FashionMNIST(data_path, train = False, 
                                            download = True, transform=dataset_transform)
    return train_dataset, test_dataset