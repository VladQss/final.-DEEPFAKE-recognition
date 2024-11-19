import kagglehub

# Download latest version
path = kagglehub.dataset_download("itamargr/dfdc-faces-of-the-train-sample")

print("Path to dataset files:", path)