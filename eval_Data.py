import logging
from pathlib import Path
import torch
import pandas as pd
from Network_structure import NeuralNetwork
from load_Data import load_data
from config import EVAL_MODEL_FILE, OUTPUT_FILE, INPUT_FEATURES

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')

def load_model(model_path: str, device: torch.device) -> NeuralNetwork:
    """
    Loads the model state from a specified file onto the given device.
    
    Args:
        model_path (str): Path to the model file.
        device (torch.device): The device to load the model on.
        
    Returns:
        NeuralNetwork: The loaded neural network in evaluation mode.
        
    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Model file '{model_path}' not found.")
    
    model = NeuralNetwork()
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

def predict_and_save_results(x: torch.Tensor, y: torch.Tensor, model: NeuralNetwork,
                             input_features: list, output_file: str) -> None:
    """
    Runs predictions using the given model, appends the predictions to the DataFrame,
    and saves the results to an Excel file.
    
    Args:
        x (torch.Tensor): The input data.
        y (torch.Tensor): The target data.
        model (NeuralNetwork): The pretrained model.
        input_features (list): List of input feature column names.
        output_file (str): Path where the result Excel file will be saved.
    """
    
    with torch.no_grad():
        outputs = model(x)

    x = pd.DataFrame(x, columns=input_features)
    x['Actual'] = y
    x['Predict'] = outputs
    x['Differ'] = 100 * abs(x['Predict'] - x['Actual']) / x['Actual']

    out_path = Path(output_file)
    x.to_excel(out_path, index=False)
    logging.info(f"Results saved to: {out_path}")

def main() -> None:
    """
    Main function to load data, model, run predictions, and save the output.
    """
    try:
        # Optionally select CUDA if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {device}")

        # Load data
        x, y = load_data()
        
        # Load pretrained model
        model = load_model(EVAL_MODEL_FILE, device)
        
        # Run predictions and save results
        predict_and_save_results(x, y, model, INPUT_FEATURES, OUTPUT_FILE)
        
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
