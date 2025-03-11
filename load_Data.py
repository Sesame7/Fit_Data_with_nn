import argparse
import logging
import os
import sys
from typing import List, Tuple

import pandas as pd
import torch

from config import INPUT_FILE, INPUT_FEATURES, OUTPUT_FEATURES  # Import constants from config

# Configure logging: You can adjust logging level as needed.
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def load_data(file_path: str = INPUT_FILE, input_features: List[str] = INPUT_FEATURES, 
              output_features: List[str] = OUTPUT_FEATURES) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Reads data from an Excel file, verifies that required features exist, and returns tensors.
    
    Args:
        file_path: Path to the Excel file.
        input_features: List of column names for input features.
        output_features: List of column names for output features.
    
    Returns:
        A tuple (inputs, outputs) as torch.Tensors.
    
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If any required feature is missing.
    """
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file '{file_path}' not found.")
    
    try:
        # Read the Excel file
        data = pd.read_excel(file_path)
    except Exception as e:
        raise ValueError(f"Failed to read Excel file '{file_path}': {e}")

    # Verify that every input feature exists
    missing_inputs = [feature for feature in input_features if feature not in data.columns]
    if missing_inputs:
        raise ValueError(f"Missing input features in data: {missing_inputs}")
    
    # Verify that every output feature exists
    missing_outputs = [feature for feature in output_features if feature not in data.columns]
    if missing_outputs:
        raise ValueError(f"Missing output features in data: {missing_outputs}")
    
    # Convert features to torch tensors
    inputs = torch.tensor(data[input_features].values, dtype=torch.float32)
    outputs = torch.tensor(data[output_features].values, dtype=torch.float32)
    
    return inputs, outputs


def main(file_path: str, input_features: List[str], output_features: List[str]) -> None:
    """
    Entry point for the program.
    
    Args:
        file_path: Path to the data file.
        input_features: List of input feature names.
        output_features: List of output feature names.
    """
    try:
        inputs, outputs = load_data(file_path, input_features, output_features)
        logging.info(f"Input tensor shape: {inputs.shape}")
        logging.info(f"Output tensor shape: {outputs.shape}")
    
    except (FileNotFoundError, ValueError) as e:
        # Log the error and exit with status 1.
        logging.error(e)
        sys.exit(1)


if __name__ == "__main__":
    # Parse command-line arguments for customization
    parser = argparse.ArgumentParser(description="Load data from an Excel file into PyTorch tensors.")
    parser.add_argument("--file", type=str, default=INPUT_FILE,
                        help="Path to the Excel file (default: value from config)")
    parser.add_argument("--input_features", type=str, nargs='+', default=INPUT_FEATURES,
                        help="List of input feature column names (default: value from config)")
    parser.add_argument("--output_features", type=str, nargs='+', default=OUTPUT_FEATURES,
                        help="List of output feature column names (default: value from config)")
                        
    args = parser.parse_args()
    main(args.file, args.input_features, args.output_features)
