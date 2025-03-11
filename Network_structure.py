import torch
import torch.nn as nn
from config import INPUT_FEATURES, OUTPUT_FEATURES, dropout_rate

class NeuralNetwork(nn.Module):
    """
    Neural network architecture for regression or similar tasks.
    
    The network:
      - Applies a logarithmic transformation on the input.
      - Processes the data through several fully connected layers with ReLU activations.
      - Uses dropout for regularization after each activation.
      - Outputs a transformation that is exponentiated.
      
    Note:
      - Adjust the dropout rate if necessary.
      - The initial log transformation includes a small constant (1e-6)
        to ensure numerical stability.
    """
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # First layer: maps input features to 128 neurons.
        self.fc1 = nn.Linear(len(INPUT_FEATURES), 128)
        # Second layer: maps 128 neurons to 64 neurons.
        self.fc2 = nn.Linear(128, 64)
        # Third layer: maps 64 neurons to 32 neurons.
        self.fc3 = nn.Linear(64, 32)
        # Output layer: maps 32 neurons to output features.
        self.fc4 = nn.Linear(32, len(OUTPUT_FEATURES))
        # Dropout layer to help with regularization (drop probability = 0.1).
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply logarithmic transformation for numerical stability.
        # The addition of 1e-6 prevents log(0) issues.
        x = torch.log1p(x + 1e-6)
        
        # First fully connected layer with ReLU activation.
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        
        # Second layer with ReLU activation.
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        
        # Third layer with ReLU activation.
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)
        
        # Final output layer.
        x = self.fc4(x)
        # Exponential transformation of the output.
        x = torch.exp(x)
        
        return x
