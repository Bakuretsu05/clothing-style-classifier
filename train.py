import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.models import MobileNet_V3_Small_Weights
import torchvision.transforms.functional as TF
from PIL import Image
import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
NUM_CLASSES = 7
INPUT_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Class labels
CLASS_NAMES = [
    'T-shirt',
    'Polo',
    'Formal_Shirt',
    'Tank_Top',
    'Sweater',
    'Hoodie',
    'Jacket'
]

# Label mapping from CSV labels to our 7 classes
LABEL_MAPPING = {
    'T-Shirt': 'T-shirt',
    'T-shirt': 'T-shirt',  # Handle both variations
    'Polo': 'Polo',
    'Shirt': 'Formal_Shirt',
    'Longsleeve': 'Formal_Shirt',
    'Blouse': 'Formal_Shirt',
    'Undershirt': 'Tank_Top',
    'Top': 'Tank_Top',
    'Body': 'Sweater',  # Assuming "Body" refers to sweater/body covering
    'Hoodie': 'Hoodie',
    'Outwear': 'Jacket',
    'Blazer': 'Jacket',
    'Jacket': 'Jacket',
}

class ClothingDataset(Dataset):
    """Custom dataset for clothing images loaded from CSV."""
    
    def __init__(self, csv_path, images_dir, split='train', transform=None, 
                 train_ratio=0.8, random_state=42):
        """
        Args:
            csv_path: Path to the CSV file with columns: image, sender_id, label, kids
            images_dir: Directory containing the image files
            split: 'train' or 'val'
            transform: Image transforms to apply
            train_ratio: Ratio of data to use for training (rest for validation)
            random_state: Random seed for train/val split
        """
        self.images_dir = Path(images_dir)
        self.transform = transform
        
        # Load CSV
        df = pd.read_csv(csv_path)
        
        # Map labels to our 7 classes
        df['mapped_label'] = df['label'].map(LABEL_MAPPING)
        
        # Filter out rows that don't map to our 7 classes (drop NaN)
        df = df.dropna(subset=['mapped_label'])
        
        # Filter out "Not sure" and "Skip" labels
        df = df[~df['label'].isin(['Not sure', 'Skip'])]
        
        # Create label to index mapping
        self.label_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}
        df['label_idx'] = df['mapped_label'].map(self.label_to_idx)
        
        # Drop any remaining rows with invalid labels
        df = df.dropna(subset=['label_idx'])
        
        # Split into train/val
        train_df, val_df = train_test_split(
            df, test_size=1-train_ratio, random_state=random_state, 
            stratify=df['label_idx']  # Stratify to maintain class distribution
        )
        
        # Select appropriate split
        if split == 'train':
            self.df = train_df.reset_index(drop=True)
        else:
            self.df = val_df.reset_index(drop=True)
        
        print(f"Loaded {len(self.df)} images for {split} split")
        print(f"Label distribution for {split}:")
        print(self.df['mapped_label'].value_counts())
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Construct image path: images_dir/{uuid}.jpg
        image_uuid = row['image']
        img_path = self.images_dir / f"{image_uuid}.jpg"
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Could not load image {img_path}: {e}")
            # Return a black image if file not found
            image = Image.new('RGB', (224, 224), color='black')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        label = int(row['label_idx'])
        
        return image, label


def get_data_transforms():
    """Define data augmentation and preprocessing transforms."""
    train_transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])  # ImageNet stats
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def create_model(num_classes=NUM_CLASSES):
    """Create MobileNetV3-Small model with pretrained weights."""
    # Load pretrained MobileNetV3-Small
    model = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    
    # Replace classifier head for our 7 classes
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier[0].in_features, 512),
        nn.Hardswish(),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes)
    )
    
    return model


def train_model(model, train_loader, val_loader, num_epochs=NUM_EPOCHS):
    """Train the model."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                     factor=0.5, patience=5, verbose=True)
    
    best_val_acc = 0.0
    train_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        
        # Validation phase
        val_acc, val_f1, _ = evaluate_model(model, val_loader)
        val_accuracies.append(val_acc)
        
        scheduler.step(epoch_loss)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Loss: {epoch_loss:.4f}, '
              f'Val Acc: {val_acc:.4f}, '
              f'Val F1: {val_f1:.4f}')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs('models', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_names': CLASS_NAMES
            }, 'models/best_model.pth')
            print(f'  -> Saved best model with val acc: {best_val_acc:.4f}')
    
    return train_losses, val_accuracies


def evaluate_model(model, data_loader):
    """Evaluate the model and return accuracy, F1-score, and predictions."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_per_class = f1_score(all_labels, all_preds, average=None)
    
    return accuracy, f1_macro, (all_labels, all_preds)


def print_classification_report(true_labels, pred_labels):
    """Print detailed classification report."""
    report = classification_report(true_labels, pred_labels, 
                                   target_names=CLASS_NAMES, 
                                   digits=4)
    print("\nClassification Report:")
    print(report)


def plot_training_history(train_losses, val_accuracies):
    """Plot training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(train_losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    ax2.plot(val_accuracies)
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Training history saved to training_history.png")


def main():
    """Main training function."""
    print(f"Using device: {DEVICE}")
    
    # Data transforms
    train_transform, val_transform = get_data_transforms()
    
    # Create datasets from CSV
    csv_path = 'data/raw/images.csv'
    images_dir = 'data/raw/images'
    
    train_dataset = ClothingDataset(
        csv_path=csv_path,
        images_dir=images_dir,
        split='train',
        transform=train_transform,
        train_ratio=0.8,
        random_state=42
    )
    val_dataset = ClothingDataset(
        csv_path=csv_path,
        images_dir=images_dir,
        split='val',
        transform=val_transform,
        train_ratio=0.8,
        random_state=42
    )
    
    if len(train_dataset) == 0:
        print("ERROR: No training images found!")
        print(f"Please check: CSV file at {csv_path} and images directory at {images_dir}")
        return
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                            shuffle=False, num_workers=4, pin_memory=True)
    
    # Create model
    model = create_model()
    model = model.to(DEVICE)
    
    print(f"Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Train model
    print("\nStarting training...")
    train_losses, val_accuracies = train_model(model, train_loader, val_loader)
    
    # Final evaluation
    print("\n" + "="*50)
    print("Final Evaluation on Validation Set:")
    print("="*50)
    final_acc, final_f1, (true_labels, pred_labels) = evaluate_model(model, val_loader)
    print(f"Accuracy: {final_acc:.4f}")
    print(f"F1-Score (macro): {final_f1:.4f}")
    
    print_classification_report(true_labels, pred_labels)
    
    # Plot training history
    plot_training_history(train_losses, val_accuracies)
    
    print("\nTraining completed!")
    print(f"Best model saved to: models/best_model.pth")


if __name__ == '__main__':
    main()
