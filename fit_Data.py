import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from Network_structure import NeuralNetwork
from load_Data import load_data
import numpy as np
from config import train_percentage, val_percentage, num_epochs, patience, batch_size, learning_rate
import os

# Configuration checks
if train_percentage < 0 or val_percentage < 0:
    raise ValueError("The train and validation percentages must be non-negative.")
if train_percentage + val_percentage >= 1.0:
    raise ValueError("The sum of train and validation percentages must be less than 1.")
if num_epochs <= 0:
    raise ValueError("The number of epochs must be positive.")
if patience <= 0:
    raise ValueError("The patience value must be positive.")

# Load data and convert to PyTorch tensors
x_tensor, y_tensor = load_data()

# Create dataset and split it into training, validation, and test sets
dataset = TensorDataset(x_tensor, y_tensor)
train_size = int(train_percentage * len(dataset))
val_size = int(val_percentage * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Data loaders with batch size 10 (could also be parameterized)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Set device to GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Initialize model and prepare it for training
model = NeuralNetwork().to(device)

# Define loss function and optimizer
criterion = nn.SmoothL1Loss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

best_val_loss = float('inf')
patience_counter = 0

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0  # For tracking training loss on current epoch
    
    for inputs, targets in train_loader:
        # Move data to the configured device
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()  # Reset gradients
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()       # Backpropagation
        optimizer.step()      # Update weights
        
        running_loss += loss.item()
    
    # Calculate average training loss for the epoch
    avg_train_loss = running_loss / len(train_loader)
    
    # Every 10 epochs, evaluate on validation set and check for early stopping
    if (epoch + 1) % 10 == 0:
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
        avg_val_loss = val_loss / len(val_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        
        # Early stopping logic: save model if validation loss improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
        
        # If no improvement for "patience" number of checks, exit training
        if patience_counter >= patience:
            print("Early stopping triggered")
            break

# Load the best saved model based on validation performance
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Evaluate model performance on the test set
test_loss = 0.0
with torch.no_grad():
    for i, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        test_loss += criterion(outputs, targets).item()
        
        # Output the first batch's data for inspection
        if i == 0:
            np.set_printoptions(precision=3, suppress=True)
            print("Inputs:\n", inputs.cpu().numpy())
            print("Targets:\n", targets.cpu().numpy())
            print("Predict:\n", outputs.cpu().numpy())
            
avg_test_loss = test_loss / len(test_loader)
print(f'Test Loss: {avg_test_loss:.3f}')

# Compare best_model.pth and model.pth on the full dataset
if not os.path.exists('model.pth'):
    torch.save(model.state_dict(), 'model.pth')
    print("model.pth did not exist and has been created.")
else:
    full_dataset = TensorDataset(x_tensor, y_tensor)
    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)

    # Load the best model for evaluation
    best_model = NeuralNetwork().to(device)
    best_model.load_state_dict(torch.load('best_model.pth'))
    best_model.eval()

    # Load the current model for evaluation
    current_model = NeuralNetwork().to(device)
    current_model.load_state_dict(torch.load('model.pth'))
    current_model.eval()

    best_loss = 0.0
    current_loss = 0.0

    with torch.no_grad():
        for inputs, targets in full_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs_best = best_model(inputs)
            outputs_current = current_model(inputs)
            best_loss += criterion(outputs_best, targets).item()
            current_loss += criterion(outputs_current, targets).item()

    best_loss /= len(full_loader)
    current_loss /= len(full_loader)

    print(f'Full dataset loss - best_model.pth: {best_loss:.4f}, model.pth: {current_loss:.4f}')

    if best_loss < current_loss:
        torch.save(best_model.state_dict(), 'model.pth')
        print("Updated model.pth with best_model.pth")
    else:
        print("model.pth remains unchanged")
