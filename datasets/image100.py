from torchvision import datasets, transforms

def create_dataset(data_path):
    dataset_transform = transforms.Compose([transforms.Resize((144,144)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                                ])
    train_dataset = datasets.ImageFolder(data_path, transform = dataset_transform)
    test_dataset = datasets.ImageFolder(data_path, transform = dataset_transform)

    return train_dataset, test_dataset