import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Constants
EPOCHS = 1000
LEARNING_RATE = 0.01
SEED = 42

# Set random seed for reproducibility
np.random.seed(SEED)
torch.manual_seed(SEED)

# Generate random cubic function coefficients
coefficients = np.random.rand(17) * 10

def random_cubic_function(x, y):
    """
    Generate a random cubic function based on the given coefficients.
    """
    return (coefficients[0] * x**-1 + coefficients[1] * y**-1 + 
            coefficients[2] * x**-1 * y**-1 + coefficients[3])

def generate_data():
    """
    Generate data for the random cubic function.
    """
    x = np.linspace(0, 2, 100) + 1
    y = np.linspace(0, 2, 100) + 1
    X, Y = np.meshgrid(x, y)
    Z = random_cubic_function(X, Y)
    return X, Y, Z

def prepare_data(X, Y, Z):
    """
    Prepare data for PyTorch.
    """
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    Z_flat = Z.flatten()
    data = np.vstack((X_flat, Y_flat)).T
    targets = Z_flat
    data_tensor = torch.tensor(data, dtype=torch.float32)
    targets_tensor = torch.tensor(targets, dtype=torch.float32).view(-1, 1)
    return data_tensor, targets_tensor

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(2, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x

def train_network(net, data_tensor, targets_tensor, epochs, learning_rate):
    """
    Train the neural network.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = net(data_tensor)
        loss = criterion(outputs, targets_tensor)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

def plot_results(X, Y, Z, predictions):
    """
    Plot the original and fitted functions.
    """
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X, Y, Z, cmap='viridis')
    ax1.set_title('Original Function')
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(X, Y, predictions, cmap='viridis')
    ax2.set_title('Fitted Function')
    plt.show()

def main():
    X, Y, Z = generate_data()
    data_tensor, targets_tensor = prepare_data(X, Y, Z)
    net = NeuralNetwork()
    train_network(net, data_tensor, targets_tensor, EPOCHS, LEARNING_RATE)
    with torch.no_grad():
        predictions = net(data_tensor).numpy().reshape(100, 100)
    plot_results(X, Y, Z, predictions)

if __name__ == "__main__":
    main()
