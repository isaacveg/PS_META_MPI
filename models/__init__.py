"""
Import every model from corresponding file, for adding models, please create model.py and add it here.
"""

from .AlexNet import *
from .VGG import *
from .LR import *
from .CNN import *
from .ResNet import *

def create_model_instance(dataset_type, model_type, class_num=10):
    if dataset_type == 'FashionMNIST':
        if model_type == 'LR':
            model = LR.MNIST_LR_Net()
        else:
            model = CNN.MNIST_Net()

    elif dataset_type == 'EMNIST':
        if model_type == 'VGG19':
            model = VGG19_EMNIST()
        if model_type == 'CNN':
            model=EMNIST_CNN()

    elif dataset_type == 'SVHN':
        if model_type == 'VGG19':
            model = VGG19_EMNIST()
        if model_type == 'CNN':
            model=EMNIST_CNN()
    
    elif dataset_type == 'CIFAR10':
        if model_type == 'AlexNet':
            model=AlexNet(class_num)
        elif model_type == 'VGG9':
            model=VGG9()
        elif model_type == 'AlexNet2':
            model=AlexNet2(class_num)
        elif model_type == 'VGG16':
            model=VGG16_Cifar10()
    
    elif dataset_type == 'CIFAR100':
        if model_type == 'ResNet':
            model = ResNet9(num_classes=100)
        elif model_type == 'VGG16':
            model = VGG16_Cifar100()
    
    elif dataset_type == 'tinyImageNet':
        if model_type == 'ResNet':
            model = ResNet50(class_num=200)
    
    elif dataset_type == 'image100':
        if model_type == 'AlexNet':
            model = AlexNet_IMAGE()
        elif model_type == 'VGG16':
            model = VGG16_IMAGE()
    
    else:
        raise ValueError('Not valid dataset')
    
    return model