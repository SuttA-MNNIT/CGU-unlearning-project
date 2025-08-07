import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

def prepare_data(forget_class, num_to_forget, sample_size, batch_size):
    """
    Loads CIFAR-10 and splits it into train, test, retain, forget, and sample sets.

    Returns:
        A tuple of DataLoaders and the list of class names.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Download and load the datasets
    trainset_full = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    class_names = trainset_full.classes
    
    # Identify indices for the forget and retain sets
    train_targets = np.array(trainset_full.targets)
    class_indices = np.where(train_targets == forget_class)[0]
    np.random.shuffle(class_indices)
    forget_indices = class_indices[:num_to_forget]
    
    all_indices = np.arange(len(trainset_full))
    retain_indices = np.setdiff1d(all_indices, forget_indices, assume_unique=True)

    # Create the datasets from subsets of the full training set
    retain_set = Subset(trainset_full, retain_indices)
    forget_set = Subset(trainset_full, forget_indices)
    
    # Create a small sample set from the retain data for Causal Tracing
    sample_indices_for_tracing = np.random.choice(retain_indices, sample_size, replace=False)
    sample_set = Subset(trainset_full, sample_indices_for_tracing)
    
    # Create DataLoaders
    train_loader_full = DataLoader(trainset_full, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    retain_loader = DataLoader(retain_set, batch_size=batch_size, shuffle=True)
    forget_loader = DataLoader(forget_set, batch_size=batch_size, shuffle=False)
    sample_loader = DataLoader(sample_set, batch_size=batch_size, shuffle=False)
    
    return train_loader_full, test_loader, retain_loader, forget_loader, sample_loader, class_names