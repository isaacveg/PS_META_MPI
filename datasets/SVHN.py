from torchvision import datasets, transforms

def create_dataset(data_path):
    dataset_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])
    train_dataset = datasets.SVHN(data_path+'/SVHN_data', split='train',
                                            download = True, transform=dataset_transform)
    test_dataset = datasets.SVHN(data_path+'/SVHN_data', split='test', 
                                            download = True, transform=dataset_transform)
    return train_dataset, test_dataset