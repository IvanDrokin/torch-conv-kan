import json
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchvision.transforms.autoaugment import AutoAugmentPolicy
from tqdm import tqdm

from models import SimpleConvKALN, SimpleFastConvKAN, SimpleConvKAN, SimpleConv, EightSimpleConvKALN, \
    EightSimpleFastConvKAN, EightSimpleConvKAN, EightSimpleConv, SimpleConvKACN, EightSimpleConvKACN, \
    SimpleConvKAGN, EightSimpleConvKAGN, SimpleConvWavKAN, EightSimpleConvWavKAN
from kan_convs import KANConv2DLayer, KALNConv2DLayer, FastKANConv2DLayer, KACNConv2DLayer, KAGNConv2DLayer, WavKANConv2DLayer


class OutputHook(list):
    """ Hook to capture module outputs.
    """
    def __call__(self, module, input, output):
        self.append(output)


class Trainer:
    def __init__(self, model_compiled, model, device, train_loader, val_loader, optimizer, scheduler, criterion,
                 l1_activation_penalty=0.0, l2_activation_penalty=0.0, is_moe=False):
        # Initialize the Trainer class with model, device, data loaders, optimizer, scheduler, and loss function
        self.model = model  # Neural network model to be trained and validated
        self.model_compiled = model_compiled
        self.device = device  # Device on which the model will be trained (e.g., 'cuda' or 'cpu')
        self.train_loader = train_loader  # DataLoader for the training dataset
        self.val_loader = val_loader  # DataLoader for the validation dataset
        self.optimizer = optimizer  # Optimizer for adjusting model parameters
        self.scheduler = scheduler  # Learning rate scheduler for the optimizer
        self.criterion = criterion  # Loss function to measure model performance
        self.l1_activation_penalty = l1_activation_penalty
        self.l2_activation_penalty = l2_activation_penalty
        self.scaler = torch.cuda.amp.GradScaler()
        self.output_hook = OutputHook()
        self.is_moe = is_moe
        for module in self.model.modules():
            if isinstance(module, (KANConv2DLayer, KALNConv2DLayer, FastKANConv2DLayer,
                                   KACNConv2DLayer, KAGNConv2DLayer, WavKANConv2DLayer)):
                module.register_forward_hook(self.output_hook)

    def train_epoch(self):
        # Train the model for one epoch and return the average loss and accuracy
        self.model.train()  # Set the model to training mode
        total_loss, total_accuracy = 0, 0  # Initialize accumulators for loss and accuracy
        for images, labels in self.train_loader:
            # Reshape images and move images and labels to the specified device
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()  # Clear previous gradients
                output = self.model_compiled(images)  # Forward pass through the model
                moe_loss = 0
                if self.is_moe:
                    output, moe_loss = output
                loss = self.criterion(output, labels) + moe_loss# Compute loss between model output and true labels

                l2_penalty = 0.
                l1_penalty = 0.
                for _output in self.output_hook:
                    if self.l1_activation_penalty > 0:
                        l1_penalty += torch.norm(_output, 1, dim=0).mean()
                    if self.l2_activation_penalty > 0:
                        l2_penalty += torch.norm(_output, 2, dim=0).mean()
                l2_penalty *= self.l2_activation_penalty
                l1_penalty *= self.l1_activation_penalty

                loss = loss + l1_penalty + l2_penalty
            self.scaler.scale(loss).backward()

            # Unscales gradients and calls
            # or skips optimizer.step()
            self.scaler.step(self.optimizer)

            # Updates the scale for next iteration
            self.scaler.update()
            # loss.backward()  # Backpropagate the loss to compute gradients
            # self.optimizer.step()  # Update model parameters
            # Calculate accuracy by comparing predicted and true labels
            accuracy = (output.argmax(dim=1) == labels).float().mean().item()
            # Accumulate total loss and accuracy
            total_loss += loss.item()
            total_accuracy += accuracy
            self.output_hook.clear()
        # Return average loss and accuracy for the epoch
        return total_loss / len(self.train_loader), total_accuracy / len(self.train_loader)

    def validate_epoch(self):
        # Validate the model for one epoch and return the average loss and accuracy
        self.model.eval()  # Set the model to evaluation mode
        val_loss, val_accuracy = 0, 0  # Initialize accumulators for validation loss and accuracy
        with torch.no_grad():  # Disable gradient computation
            for images, labels in self.val_loader:
                # Reshape images and move images and labels to the specified device
                images, labels = images.to(self.device), labels.to(self.device)
                if self.is_moe:
                    output, _ = self.model(images, train=False)  # Forward pass through the model
                else:
                    output = self.model(images)
                # Accumulate validation loss and accuracy
                val_loss += self.criterion(output, labels).item()
                val_accuracy += (output.argmax(dim=1) == labels).float().mean().item()
        # Return average validation loss and accuracy for the epoch
        return val_loss / len(self.val_loader), val_accuracy / len(self.val_loader)

    def fit(self, epochs):
        # Train and validate the model over multiple epochs
        train_accuracies, val_accuracies = [], []  # Lists to store accuracies for each epoch
        pbar = tqdm(range(epochs), desc="Epoch Progress")  # Progress bar to track training progress
        for epoch in pbar:
            # Train and validate for one epoch
            train_loss, train_accuracy = self.train_epoch()
            val_loss, val_accuracy = self.validate_epoch()
            # Log metrics to Weights & Biases
            # Update progress bar with current epoch loss and accuracy
            pbar.set_description(f"Epoch {epoch + 1} | Train Loss: {train_loss:.4f} | Val Accuracy: {val_accuracy:.4f}")
            self.scheduler.step()  # Update learning rate based on the scheduler
            # Store train and validation accuracies
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)
        return train_accuracies, val_accuracies


def quantize_and_evaluate(model, val_loader, criterion, save_path):
    # Function to quantize the model, evaluate its performance, and save it
    model.cpu()  # Ensure the model is on the CPU for quantization
    # Quantize the model to reduce size and potentially speed up inference
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Conv2d},  # Specify which layers to quantize
        dtype=torch.qint8  # Set the data type for quantized weights
    )
    quantized_model.eval()  # Set the quantized model to evaluation mode
    quantized_val_loss, quantized_val_accuracy = 0, 0  # Initialize accumulators for loss and accuracy
    start_time = time.time()  # Record the start time for evaluation
    with torch.no_grad():  # Disable gradient computation
        for images, labels in val_loader:
            # Reshape images and move images and labels to the CPU
            images, labels = images.to(torch.device('cpu')), labels.to(torch.device('cpu'))
            output = quantized_model(images)  # Forward pass through the quantized model
            if isinstance(output, tuple):
                output = output[0]
            # Accumulate validation loss and accuracy for the quantized model
            quantized_val_loss += criterion(output, labels).item()
            quantized_val_accuracy += (output.argmax(dim=1) == labels).float().mean().item()
    evaluation_time = time.time() - start_time  # Calculate total evaluation time

    # Create directories if necessary and save the quantized model
    # os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # torch.save(quantized_model.state_dict(), save_path)

    return quantized_val_loss / len(val_loader), quantized_val_accuracy / len(val_loader), evaluation_time


def train_and_validate(model, bs, epochs=15, dataset_name='MNIST', model_save_dir="./models",
                       l1_activation_penalty=0.0, l2_activation_penalty=0.0, is_moe=False
                       ):
    # Function to train, validate, quantize the model, and evaluate the quantized model
    # Define the transformations for the datasets
    # Load and transform the MNIST training dataset
    if dataset_name == 'MNIST':
        transform_train = v2.Compose([
            v2.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            v2.ColorJitter(brightness=0.2, contrast=0.2),
            v2.ToTensor(),
            v2.Normalize((0.5,), (0.5,))
        ])

        transform_test = v2.Compose([
            v2.ToTensor(),
            v2.Normalize((0.5,), (0.5,))
        ])
        trainset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform_train)
        # Load and transform the MNIST validation dataset
        valset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform_test)
        # Create DataLoaders for training and validation datasets
    elif dataset_name == 'CIFAR10':
        transform_train = v2.Compose([

            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomChoice([v2.AutoAugment(AutoAugmentPolicy.CIFAR10),
                             v2.AutoAugment(AutoAugmentPolicy.IMAGENET),
                             v2.AutoAugment(AutoAugmentPolicy.SVHN),
                             v2.TrivialAugmentWide()]),
            v2.ToTensor(),
            v2.Normalize((0.5,), (0.5,))
        ])
        transform_test = v2.Compose([
            v2.ToTensor(),
            v2.Normalize((0.5,), (0.5,))
        ])
        trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
        # Load and transform the CIFAR10 validation dataset
        valset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)
        # Create DataLoaders for training and validation datasets
    else:
        transform_train = v2.Compose([

            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomChoice([v2.AutoAugment(AutoAugmentPolicy.CIFAR10),
                             v2.AutoAugment(AutoAugmentPolicy.IMAGENET),
                             v2.AutoAugment(AutoAugmentPolicy.SVHN),
                             v2.TrivialAugmentWide()]),
            v2.ToTensor(),
            v2.Normalize((0.5,), (0.5,))
        ])
        transform_test = v2.Compose([
            v2.ToTensor(),
            v2.Normalize((0.5,), (0.5,))
        ])
        trainset = torchvision.datasets.CIFAR100(root="./data", train=True, download=True, transform=transform_train)
        # Load and transform the CIFAR100 validation dataset
        valset = torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=transform_test)
        # Create DataLoaders for training and validation datasets
    trainloader = DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=12)
    valloader = DataLoader(valset, batch_size=bs, shuffle=False, num_workers=12)

    # Determine the appropriate device based on GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # Move the model to the selected device

    # model_compiled = torch.compile(model)

    # Set up the optimizer with specified parameters
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # Define the learning rate scheduler
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.975)
    # Set the loss function for training and validation
    criterion = nn.CrossEntropyLoss()

    # Initialize the Trainer and train the model
    trainer = Trainer(model, model, device, trainloader, valloader, optimizer, scheduler, criterion,
                      l1_activation_penalty=l1_activation_penalty, l2_activation_penalty=l2_activation_penalty,
                      is_moe=is_moe)
    train_accuracies, val_accuracies = trainer.fit(epochs)

    # Ensure the directory for saving models exists
    os.makedirs(model_save_dir, exist_ok=True)

    # Save the trained model's state dictionary
    # torch.save(model.state_dict(), os.path.join(model_save_dir, "original_model.pth"))

    # Quantize and evaluate the quantized model
    quantized_loss, quantized_accuracy, quantized_time = quantize_and_evaluate(model, valloader, criterion,
                                                                               os.path.join(model_save_dir,
                                                                                            "quantized_model.pth"))
    print(
        f"Quantized Model - Validation Loss: {quantized_loss:.4f}, Validation Accuracy: {quantized_accuracy:.4f}, Evaluation Time: {quantized_time:.4f} seconds")

    # Evaluate the time taken to evaluate the original model
    model.eval().to(device)
    start_time = time.time()
    with torch.no_grad():
        for images, labels in valloader:
            # Reshape images and move them and labels to the selected device
            images, labels = images.to(device), labels.to(device)
            output = model(images)
    original_time = time.time() - start_time  # Calculate the total evaluation time

    # Print the results summary
    print(f"Original Model Evaluation Time: {original_time:.4f} seconds")
    print(f"Train Accuracies: {train_accuracies}")
    print(f"Validation Accuracies: {val_accuracies}")
    report = {"Validation Accuracies": val_accuracies, 'Train Accuracies': train_accuracies,
              "Validation Accuracy - q": quantized_accuracy, 'Evaluation Time - q': quantized_time,
              'Evaluation Time': original_time, 'Parameters': count_parameters(model)}
    with open(os.path.join(model_save_dir, 'report.json'), 'w') as f:
        json.dump(report, f)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_kan_model(num_classes, input_channels):
    return SimpleConvKAN([8 * 4, 16 * 4, 32 * 4, 64 * 4], num_classes=num_classes, input_channels=input_channels,
                         spline_order=3, groups=1, dropout=0.25, dropout_linear=0.5, l1_penalty=0.00000,
                         degree_out=1)


def get_kaln_model(num_classes, input_channels):
    return SimpleConvKALN([8 * 4, 16 * 4, 32 * 4, 64 * 4], num_classes=num_classes, input_channels=input_channels,
                          degree=3, groups=1, dropout=0.25, dropout_linear=0.5, l1_penalty=0.00000,
                          degree_out=1)


def get_kagn_model(num_classes, input_channels):
    return SimpleConvKAGN([8 * 4, 16 * 4, 32 * 4, 64 * 4], num_classes=num_classes, input_channels=input_channels,
                          degree=3, groups=4, dropout=0.25, dropout_linear=0.5, l1_penalty=0.00000,
                          degree_out=1)


def get_kacn_model(num_classes, input_channels):
    return SimpleConvKACN([8 * 4, 16 * 4, 32 * 4, 64 * 4], num_classes=num_classes, input_channels=input_channels,
                          degree=6, groups=4, dropout=0.25, dropout_linear=0.5, l1_penalty=0.00000,
                          degree_out=1)


def get_wavkan_model(num_classes, input_channels):
    return SimpleConvWavKAN([8 * 4, 16 * 4, 32 * 4, 64 * 4], num_classes=num_classes, input_channels=input_channels,
                            wavelet_type='mexican_hat', groups=1, dropout=0.25, dropout_linear=0.5, l1_penalty=0.00000,
                            degree_out=1)


def get_fast_kan_model(num_classes, input_channels):
    return SimpleFastConvKAN([8 * 4, 16 * 4, 32 * 4, 64 * 4], num_classes=num_classes, input_channels=input_channels,
                             grid_size=8, groups=1, dropout=0.25, dropout_linear=0.5, l1_penalty=0.00000,
                             degree_out=1)


def get_simple_conv_model(num_classes, input_channels):
    return SimpleConv([8 * 4, 16 * 4, 32 * 4, 64 * 4], num_classes=num_classes, input_channels=input_channels, groups=4)


def get_8kan_model(num_classes, input_channels):
    return EightSimpleConvKAN([8 * 2, 16 * 2, 32 * 2, 64 * 2, 128 * 2, 128 * 2, 128 * 4, 128 * 4],
                              num_classes=num_classes, input_channels=input_channels,
                              spline_order=3, groups=1, dropout=0.25, dropout_linear=0.5, l1_penalty=0.000000,
                              degree_out=1)


def get_8kaln_model(num_classes, input_channels):
    return EightSimpleConvKALN([8 * 2, 16 * 2, 32 * 2, 64 * 2, 128 * 2, 128 * 2, 128 * 4, 128 * 4],
                               num_classes=num_classes, input_channels=input_channels,
                               degree=3, groups=1, dropout=0.25, dropout_linear=0.5, l1_penalty=0.00000,
                               degree_out=1)


def get_8kagn_model(num_classes, input_channels):
    return EightSimpleConvKAGN([8 * 2, 16 * 2, 32 * 2, 64 * 2, 128 * 2, 128 * 2, 128 * 4, 128 * 4],
                               num_classes=num_classes, input_channels=input_channels,
                               degree=3, groups=1, dropout=0.25, dropout_linear=0.5, l1_penalty=0.00000,
                               degree_out=1)


def get_8wavkan_model(num_classes, input_channels):
    return EightSimpleConvWavKAN([8 * 2, 16 * 2, 32 * 2, 64 * 2, 128 * 2, 128 * 2, 128 * 4, 128 * 4],
                                 num_classes=num_classes, input_channels=input_channels,
                                 wavelet_type='mexican_hat', groups=1,
                                 dropout=0.25, dropout_linear=0.5, l1_penalty=0.00000,
                                 degree_out=1)


def get_8kacn_model(num_classes, input_channels):
    return EightSimpleConvKACN([8 * 2, 16 * 2, 32 * 2, 64 * 2, 128 * 2, 128 * 2, 128 * 4, 128 * 4],
                               num_classes=num_classes, input_channels=input_channels,
                               degree=3, groups=1, dropout=0.25, dropout_linear=0.5, l1_penalty=0.00000,
                               degree_out=1)


def get_8fast_kan_model(num_classes, input_channels):
    return EightSimpleFastConvKAN([8 * 2, 16 * 2, 32 * 2, 64 * 2, 128 * 2, 128 * 2, 128 * 4, 128 * 4],
                                  num_classes=num_classes, input_channels=input_channels,
                                  grid_size=8, groups=1, dropout=0.25, dropout_linear=0.5, l1_penalty=0.00000,
                                  degree_out=1)


def get_8simple_conv_model(num_classes, input_channels):
    return EightSimpleConv([8 * 2, 16 * 2, 32 * 2, 64 * 2, 128 * 2, 128 * 2, 128 * 4, 128 * 4], num_classes=num_classes,
                           input_channels=input_channels, groups=4)


if __name__ == '__main__':
    for dataset_name in ['MNIST', ]:
        # for dataset_name in ['MNIST', 'CIFAR10', 'CIFAR100']:
        #     for model_name in ['WavKAN8', 'KAN', "KALN", "FastKAN", 'KACN', 'KAGN', 'WavKAN', "Vanilla",
        #                        'KAN8', "KALN8", "FastKAN8", "KACN8", 'KAGN8', "Vanilla8"]:
        #         for dataset_name in ['MNIST', 'CIFAR10', 'CIFAR100']:
        for model_name in ['KACN', ]:
            folder_to_save = os.path.join('experiments_v3', '_'.join([model_name.lower(), dataset_name.lower()]))
            num_classes = 100 if dataset_name == 'CIFAR100' else 10
            input_channels = 1 if dataset_name == 'MNIST' else 3
            bs = 64 if model_name in ['WavKAN', 'WavKAN8'] else 128
            if model_name == 'KAN':
                kan_model = get_kan_model(num_classes, input_channels)
            elif model_name == 'KALN':
                kan_model = get_kaln_model(num_classes, input_channels)
            elif model_name == 'KAGN':
                kan_model = get_kagn_model(num_classes, input_channels)
            elif model_name == 'KACN':
                kan_model = get_kacn_model(num_classes, input_channels)
            elif model_name == 'FastKAN':
                kan_model = get_fast_kan_model(num_classes, input_channels)
            elif model_name == 'WavKAN':
                kan_model = get_wavkan_model(num_classes, input_channels)
            elif model_name == 'KAN8':
                kan_model = get_8kan_model(num_classes, input_channels)
            elif model_name == 'KALN8':
                kan_model = get_8kaln_model(num_classes, input_channels)
            elif model_name == 'KAGN8':
                kan_model = get_8kagn_model(num_classes, input_channels)
            elif model_name == 'KACN8':
                kan_model = get_8kacn_model(num_classes, input_channels)
            elif model_name == 'FastKAN8':
                kan_model = get_8fast_kan_model(num_classes, input_channels)
            elif model_name == 'WavKAN8':
                kan_model = get_8wavkan_model(num_classes, input_channels)
            elif model_name == 'Vanilla':
                kan_model = get_simple_conv_model(num_classes, input_channels)
            else:
                kan_model = get_8simple_conv_model(num_classes, input_channels)
            train_and_validate(kan_model, bs, epochs=150,
                               dataset_name=dataset_name,
                               model_save_dir=folder_to_save)  # Call the function to train and evaluate the model
