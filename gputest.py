import torch

# Check if CUDA is available
print(f"CUDA available: {torch.cuda.is_available()}")

# Get the current device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Current device: {device}")

# Get the name of the GPU
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(device)
    print(f"GPU name: {gpu_name}")
else:
    print("No GPU available, using CPU")