from torchvision import datasets, transforms

def create_dataset(data_path):
    dataset_transform =  transforms.Compose([
                           transforms.Resize(28),
                           #transforms.CenterCrop(227),
                           transforms.Grayscale(1),
                           transforms.ToTensor(),
                          transforms.Normalize((0.1307,), (0.3081,))  
                        ])
    train_dataset = datasets.EMNIST(data_path, split = 'byclass', train = True, download = True, transform=dataset_transform)
    test_dataset = datasets.EMNIST(data_path, split = 'byclass', train = False, transform=dataset_transform)
    return train_dataset, test_dataset