
# IMPORTING ALL REQUIRED LIBRARIES


import os                      # Used for directory creation and file handling
import torch                   # Core PyTorch library
import timm                    # PyTorch Image Models (pretrained CNNs)
import wandb                   # Weights & Biases for experiment tracking
import torch.nn as nn          # Neural network layers and loss functions
import torch.optim as optim    # Optimization algorithms
from torchvision import datasets, transforms  # Dataset loader & image transforms
from torch.utils.data import DataLoader       # Efficient data batching



# INITIALIZE WANDB (EXPERIMENT TRACKING)


# WandB is used to log loss and accuracy during training.
# Offline mode means logs are stored locally (no internet required).
wandb.init(
    project="human-animal-classifier",
    mode="offline"
)



# IMAGE PREPROCESSING & AUGMENTATION


# This transformation pipeline prepares images for ResNet18
# and improves generalization using data augmentation.
transform = transforms.Compose([

    transforms.Resize((224, 224)),  
    # ResNet18 expects input images of size 224x224

    transforms.RandomHorizontalFlip(),  
    # Randomly flips images horizontally to prevent overfitting

    transforms.ToTensor(),  
    # Converts PIL Image into PyTorch Tensor (C × H × W)

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        # Mean values of ImageNet dataset

        std=[0.229, 0.224, 0.225]
        # Standard deviation of ImageNet dataset
    )
])



# LOAD DATASET FROM DIRECTORY


# Dataset path containing two folders: human/ and animal/
DATASET_PATH = r"E:\project\datasets\classifier"

# ImageFolder automatically:
# - Reads images from folders
# - Assigns labels based on folder names
dataset = datasets.ImageFolder(
    DATASET_PATH,
    transform=transform
)

# DataLoader:
# - Loads data in batches
# - Shuffles data to improve training
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)

# Display class-to-index mapping
print("Class mapping:", dataset.class_to_idx)
# Example: {'human': 0, 'animal': 1}



# HANDLE CLASS IMBALANCE


# Count number of images in each class
class_counts = [0, 0]

for _, label in dataset:
    class_counts[label] += 1

# Compute inverse frequency weights
# Smaller class gets higher weight
weights = torch.tensor(
    [1.0 / class_counts[0], 1.0 / class_counts[1]],
    dtype=torch.float
)

# CrossEntropyLoss with class weights
# This prevents bias toward majority class
criterion = nn.CrossEntropyLoss(weight=weights)



# LOAD PRETRAINED RESNET18 MODEL


# timm provides pretrained CNN models
# We load ResNet18 trained on ImageNet
# num_classes=2 modifies the final layer for binary classification
model = timm.create_model(
    "resnet18",
    pretrained=True,
    num_classes=2
)



# OPTIMIZER CONFIGURATION


# Adam optimizer is used for:
# - Faster convergence
# - Adaptive learning rates
optimizer = optim.Adam(
    model.parameters(),
    lr=1e-4
)


# TRAINING LOOP


EPOCHS = 10
print("🚀 Training started...")

for epoch in range(EPOCHS):

    model.train()  
    # Sets model to training mode (enables dropout, batchnorm updates)

    total_loss = 0
    correct = 0

    for imgs, labels in loader:

        # Forward pass: input images → model → predictions
        outputs = model(imgs)

        # Compute loss between predictions and ground truth
        loss = criterion(outputs, labels)

        # Clear old gradients
        optimizer.zero_grad()

        # Backpropagation: compute gradients
        loss.backward()

        # Update model weights
        optimizer.step()

        # Accumulate loss
        total_loss += loss.item()

        # Count correct predictions
        correct += (outputs.argmax(1) == labels).sum().item()

    # Calculate accuracy for entire dataset
    acc = correct / len(dataset)

    # Log metrics to WandB
    wandb.log({
        "loss": total_loss,
        "accuracy": acc
    })

    # Print epoch summary
    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Loss: {total_loss:.2f} "
        f"Acc: {acc:.3f}"
    )



# SAVE TRAINED MODEL


# Create directory if it doesn't exist
os.makedirs(
    "E:/project/models/classifier",
    exist_ok=True
)

# Save only model weights (recommended practice)
torch.save(
    model.state_dict(),
    "E:/project/models/classifier/resnet_human_animal.pth"
)

print("Classifier trained & saved successfully...")
