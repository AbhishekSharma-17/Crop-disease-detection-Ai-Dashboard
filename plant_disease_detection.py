import multiprocessing
if __name__ == '__main__':
    # Freeze support for Windows platforms (required for multiprocessing)
    multiprocessing.freeze_support()

    import os
    import sys
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    import torchvision.models as models
    from tqdm import tqdm

    from dataset import PlantDiseaseDataset  # Import the PlantDiseaseDataset class

    # Set device to use GPU if available, otherwise use CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    sys.stdout.flush()

    # Define data transformations (resizing, normalization)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to 256x256
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Get the current script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Load datasets
    train_dataset = PlantDiseaseDataset(root_dir=os.path.join(script_dir, 'train'), transform=transform)
    val_dataset = PlantDiseaseDataset(root_dir=os.path.join(script_dir, 'valid'), transform=transform)
    test_dataset = PlantDiseaseDataset(root_dir=os.path.join(script_dir, 'test'), transform=transform)

    # Create data loaders
    batch_size = 16  # Reduced batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Load the pre-trained ResNet-18 model
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_features = model.fc.in_features
    num_classes = max(train_dataset.labels) + 1  # Assuming labels start from 0
    model.fc = nn.Linear(num_features, num_classes)  # Replace the final layer for classification
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Enable mixed precision training if supported
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    # Set the number of batches to accumulate gradients
    accumulation_steps = 4

    # Training loop
    epochs = 10
    for epoch in range(epochs):
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', unit='batch', leave=False)
        for i, (inputs, labels) in enumerate(progress_bar):
            inputs, labels = inputs.to(device), labels.to(device)

            # Mixed precision training
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            # Backward pass and optimization
            scaler.scale(loss / accumulation_steps).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item()

            # Update progress bar with current loss
            progress_bar.set_postfix(loss=running_loss / (progress_bar.n + 1))

        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}')
        sys.stdout.flush()

    # Save the trained model
    model_path = os.path.join(script_dir, "models", "plant_disease_model.pth")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f'Trained model saved to {model_path}')

    # Evaluation on test set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    sys.stdout.flush()