# Default configuration: file path and features
INPUT_FILE = './Data.xlsx'
INPUT_FEATURES = ['X1', 'X2', 'X4', 'X5', 'X6', 'X8']
OUTPUT_FEATURES = ['Y2']
OUTPUT_FILE = "Results.xlsx"
EVAL_MODEL_FILE = "model.pth"

# Model configuration
dropout_rate = 0.001
train_percentage = 0.6
val_percentage = 0.2
num_epochs = 10000
patience = 20
batch_size = 10
learning_rate = 0.00001
