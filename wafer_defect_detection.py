! kaggle datasets download qingyi/wm811k-wafer-map
! unzip wm811k-wafer-map.zip

import numpy as np
import pandas as pd
import cv2
import imgaug.augmenters as iaa
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*FixedFormatter.*")

#####################################
# Data Loading & Preprocessing Class
#####################################
class WaferDataLoader:
    def __init__(self, file_path):
        # Load the dataset from a pickle file
        self.df = pd.read_pickle(file_path)
        self.process_data()

    def process_data(self):
        """Map categorical labels to numerical values and compute wafer dimensions."""
        mapping_type = {
            'Center': 0, 'Donut': 1, 'Edge-Loc': 2, 'Edge-Ring': 3,
            'Loc': 4, 'Random': 5, 'Scratch': 6, 'Near-full': 7, 'none': 8
        }
        mapping_traintest = {'Training': 0, 'Test': 1}
        self.df.replace({'failureType': mapping_type, 'trianTestLabel': mapping_traintest}, inplace=True)
        # Compute the dimensions of each wafer map image
        self.df['waferMapDim'] = self.df.waferMap.apply(lambda x: (x.shape[0], x.shape[1]))

    def get_filtered_data(self):
        """
        Filter data into labeled and pattern-specific subsets.
        Returns:
            df_with_label: All rows with a label (0 to 8)
            df_with_pattern: Rows with actual defect patterns (0 to 7)
            df_non_pattern: Rows labeled as 'none' (8)
        """
        df_with_label = self.df[(self.df['failureType'] >= 0) & (self.df['failureType'] <= 8)].reset_index(drop=True)
        df_with_pattern = self.df[(self.df['failureType'] >= 0) & (self.df['failureType'] <= 7)].reset_index(drop=True)
        df_non_pattern = self.df[self.df['failureType'] == 8]
        return df_with_label, df_with_pattern, df_non_pattern

#####################################
# Image Processing Class
#####################################
class WaferImageProcessor:
    def __init__(self, dsize=(224, 224)):
        self.dsize = dsize
        # Define augmentation pipeline
        self.augmentor = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                rotate=(-180, 180),
                shear=(-8, 8)
            )
        ], random_order=True)

    def resize_images(self, images):
        """Resize a list or series of images to the desired dimensions."""
        return np.array([cv2.resize(img, self.dsize, interpolation=cv2.INTER_AREA) for img in images])

    def augment_images(self, images):
        """Apply augmentation to images."""
        return self.augmentor(images=images)

    def reshape_images(self, images):
        """
        Reshape images to ensure they are in RGB format.
        If an image is grayscale, convert it to 3 channels.
        """
        reshaped = []
        for img in images:
            if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[-1] == 1):
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            reshaped.append(img)
        return np.array(reshaped)

#####################################
# PyTorch Dataset for Wafer Images
#####################################
class WaferDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        """
        images: NumPy array of images with shape (N, H, W, C)
        labels: One-hot labels; will be converted to class indices.
        transform: Any additional PyTorch transforms to apply.
        """
        self.images = images
        self.labels = np.argmax(labels, axis=1)  # convert one-hot to indices
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Convert image from HxWxC (numpy) to CxHxW (PyTorch) and then to tensor
        image = self.images[idx].transpose(2, 0, 1).astype(np.float32) / 255.0
        label = self.labels[idx]
        image_tensor = torch.tensor(image)
        label_tensor = torch.tensor(label, dtype=torch.long)
        if self.transform:
            image_tensor = self.transform(image_tensor)
        return image_tensor, label_tensor

#####################################
# Data Preparation Function
#####################################
def prepare_data(df, count_per_class, processor, num_classes=9):
    """
    For each defect class, sample count_per_class images, process, augment and reshape them.
    Returns:
        x_data: Array of processed images.
        y_data: Array of one-hot encoded labels.
    """
    x_data, y_data = [], []
    for failure_num in range(num_classes):
        # Sample images for a specific defect class
        extracted = df[df['failureType'] == failure_num].sample(count_per_class, replace=True).waferMap
        # Resize images
        resized = processor.resize_images(extracted)
        # Augment images
        augmented = processor.augment_images(resized)
        # Ensure images are in RGB format
        reshaped = processor.reshape_images(augmented)
        # Create one-hot labels for this class
        labels = np.zeros((count_per_class, num_classes))
        labels[:, failure_num] = 1
        x_data.extend(reshaped)
        y_data.extend(labels)
    return np.array(x_data), np.array(y_data)

#####################################
# Transformer Model for Wafer Defect Detection
#####################################
class WaferDefectTransformer:
    def __init__(self, num_classes=9, model_name='vit_base_patch16_224'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Create a pretrained Vision Transformer from timm
        self.model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        self.model.to(self.device)
        # Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)

    def train(self, train_loader, val_loader, epochs=5):
        train_losses, val_losses = [], []
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            print(f"Epoch {epoch+1}/{epochs} starting...")
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                # Print progress every 10 batches
                if (batch_idx + 1) % 10 == 0:
                    current_loss = running_loss / total
                    current_acc = correct / total
                    print(f"Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {current_loss:.4f} | Accuracy: {current_acc:.4f}")

                    # Print GPU memory usage if GPU is available
                    if torch.cuda.is_available():
                        allocated = torch.cuda.memory_allocated(self.device) / (1024*1024)
                        reserved = torch.cuda.memory_reserved(self.device) / (1024*1024)
                        print(f"GPU Memory - Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")

            epoch_loss = running_loss / total
            epoch_acc = correct / total
            train_losses.append(epoch_loss)

            # Validation Loop
            self.model.eval()
            val_running_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    val_running_loss += loss.item() * images.size(0)
                    _, preds = torch.max(outputs, 1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)
            val_loss = val_running_loss / val_total
            val_acc = val_correct / val_total
            val_losses.append(val_loss)

            print(f"Epoch {epoch+1} complete. Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f} || Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}\n")

        return train_losses, val_losses

    def predict(self, loader):
        self.model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
        return np.array(all_preds), np.array(all_labels)

#####################################
# Utility Function: Plot Confusion Matrix using Matplotlib
#####################################
def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Annotate each cell with its count
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

#####################################
# Main Execution
#####################################
if __name__ == "__main__":
    # File path to your pickle file
    file_path = '/content/LSWMD.pkl'
    loader = WaferDataLoader(file_path)
    df_with_label, _, _ = loader.get_filtered_data()

    # Initialize image processor and prepare training and validation data
    processor = WaferImageProcessor(dsize=(224, 224))
    count_per_class_train = 80
    count_per_class_val = 20
    x_train_np, y_train_np = prepare_data(df_with_label, count_per_class_train, processor)
    x_val_np, y_val_np = prepare_data(df_with_label, count_per_class_val, processor)

    # Create PyTorch Datasets and DataLoaders
    train_dataset = WaferDataset(x_train_np, y_train_np)
    val_dataset = WaferDataset(x_val_np, y_val_np)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Initialize the Vision Transformer model for wafer defect detection
    transformer_model = WaferDefectTransformer(num_classes=9, model_name='vit_base_patch16_224')

    # Train the model
    train_losses, val_losses = transformer_model.train(train_loader, val_loader, epochs=5)

    # Plot training and validation loss
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title("Training & Validation Loss")
    plt.show()

    # Evaluate on validation set
    preds, true_labels = transformer_model.predict(val_loader)
    print(classification_report(true_labels, preds, target_names=[
          'Center','Donut','Edge-Loc','Edge-Ring','Loc','Random','Scratch','Near-full','None']))

    # Compute and plot confusion matrix using matplotlib
    cm = confusion_matrix(true_labels, preds)
    class_names = ['Center','Donut','Edge-Loc','Edge-Ring','Loc','Random','Scratch','Near-full','None']
    plot_confusion_matrix(cm, classes=class_names)