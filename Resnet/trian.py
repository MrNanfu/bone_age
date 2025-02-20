import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Custom Dataset for Bone Age Images
class BoneAgeDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Define the model (using ResNet50 as base)
class BoneAgeModel(nn.Module):
    def __init__(self):
        super(BoneAgeModel, self).__init__()
        self.base_model = resnet50(pretrained=True)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, 1)  # Predict a single value (bone age)

    def forward(self, x):
        return self.base_model(x)

# Define data preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# Function to traverse all images in a folder
def get_image_paths(folder, supported_formats=(".bmp", ".jpg", ".png")):
    image_paths = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(supported_formats):
                image_paths.append(os.path.join(root, file))
    return image_paths

# Function to extract numeric part from filename
def extract_number(filename):
    import re
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else None

# Main function to map images to RC Bone Age
def map_images_to_rc_bone_age(image_folder, excel_file, output_file):
    # Read the Excel file
    df = pd.read_excel(excel_file)

    # Ensure necessary columns exist
    if '编号' not in df.columns or 'RC骨龄' not in df.columns:
        raise ValueError("The Excel file must contain '编号' and 'RC骨龄' columns.")

    # Traverse images and map to RC Bone Age
    image_paths = get_image_paths(image_folder)
    results = []

    for path in image_paths:
        filename = os.path.basename(path)
        number = extract_number(filename)
        if number is not None:
            # Search for the number in the '编号' column
            match = df[df['编号'] == number]
            if not match.empty:
                rc_bone_age = match['RC骨龄'].values[0]
                results.append({'Image Path': path, '编号': number, 'RC骨龄': rc_bone_age})

    # Convert results to DataFrame and save
    result_df = pd.DataFrame(results)
    result_df.to_excel(output_file, index=False)
    print(f"Results saved to {output_file}")

# Function to read image paths and labels from output file
def read_image_paths_and_labels(output_file):
    df = pd.read_excel(output_file)
    if 'Image Path' not in df.columns or 'RC骨龄' not in df.columns:
        raise ValueError("The output file must contain 'Image Path' and 'RC骨龄' columns.")

    # Filter rows where 'RC骨龄' is not null
    filtered_df = df[df['RC骨龄'].notnull()]
    image_paths = filtered_df['Image Path'].tolist()
    labels = filtered_df['RC骨龄'].tolist()

    return image_paths, labels

# # Parameters (update with your actual paths)
# image_folder = '/private/workspace/cyt/child_bone/data/bone'
# excel_file = '/private/workspace/cyt/child_bone/data/data_all.xlsx'
output_file = 'output_file.xlsx'

# # Run the mapping function
# map_images_to_rc_bone_age(image_folder, excel_file, output_file)

# Example paths and labels (replace with your actual data)
image_paths, labels = read_image_paths_and_labels(output_file)

# Split dataset into training and validation sets
train_image_paths, val_image_paths, train_labels, val_labels = train_test_split(
    image_paths, labels, test_size=0.2, random_state=42
)

# Create training and validation datasets
train_dataset = BoneAgeDataset(train_image_paths, train_labels, transform=transform)
val_dataset = BoneAgeDataset(val_image_paths, val_labels, transform=transform)

# Create DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

model = BoneAgeModel()

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Modified training loop with validation
def train_model_with_validation(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        
        # Training loop
        for images, labels in train_dataloader:
            images, labels = images.to(torch.device('cuda')).float(), labels.to(torch.device('cuda')).float()
            labels = labels.unsqueeze(1)  # Ensure labels have the correct shape

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validation loop
        model.eval()
        val_predictions = []
        val_ground_truths = []
        with torch.no_grad():
            for images, labels in val_dataloader:
                images, labels = images.to(torch.device('cuda')).float(), labels.to(torch.device('cuda')).float()
                outputs = model(images)
                val_predictions.extend(outputs.cpu().numpy().flatten())
                val_ground_truths.extend(labels.cpu().numpy().flatten())

        # Compute metrics
        val_mse = mean_squared_error(val_ground_truths, val_predictions)
        val_mae = mean_absolute_error(val_ground_truths, val_predictions)

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_dataloader):.4f}, "
              f"Val MSE: {val_mse:.4f}, Val MAE: {val_mae:.4f}")
        model.train()  # Switch back to training mode

# Train and validate the model
if torch.cuda.is_available():
    model = model.to(torch.device('cuda'))
train_model_with_validation(model, train_dataloader, val_dataloader, criterion, optimizer)

