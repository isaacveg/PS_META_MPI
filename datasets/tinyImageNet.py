from torchvision import datasets, transforms

def create_dataset(data_path):
    dataset_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ])
    train_dataset = datasets.ImageFolder(data_path+'/tiny-imagenet-200/train', transform = dataset_transform)
    test_dataset = datasets.ImageFolder(data_path+'/tiny-imagenet-200/val', transform = dataset_transform)
    return train_dataset, test_dataset