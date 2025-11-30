import kagglehub

# Download latest version
path = kagglehub.dataset_download("paperxd/all-computer-prices")

print("Path to dataset files:", path)