import numpy as np
import gzip
import os
import requests

def convert_mnist_local():
    """Converts locally downloaded MNIST dataset to float32 binary files."""
    
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz",
    }
    
    input_dir = "mnist_data"

    if not os.path.exists(input_dir):
         print(f"Error: Directory '{input_dir}' not found.")
         print("Please create the directory and manually download the files into it.")
         return

    for key, filename in files.items():
        filepath = os.path.join(input_dir, filename)
        if not os.path.exists(filepath):
            print(f"Error: File not found - {filepath}")
            print("Please make sure you have downloaded all 4 files into the 'mnist_data' directory.")
            return

    print("Starting file conversion...")
    for key, filename in files.items():
        filepath = os.path.join(input_dir, filename)
        
        try:
            with gzip.open(filepath, 'rb') as f:
                if 'images' in key:
                    data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
                    num_images = data.shape[0] // (28 * 28)
                    images = data.reshape(num_images, 28 * 28).astype(np.float32)
                    images /= 255.0
                    output_path = os.path.join(input_dir, f"{key}.bin")
                    print(f"Saving {key} to {output_path} ({images.shape})")
                    images.tofile(output_path)
                else:
                    labels = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
                    one_hot_labels = np.zeros((labels.size, 10), dtype=np.float32)
                    one_hot_labels[np.arange(labels.size), labels] = 1.0
                    output_path = os.path.join(input_dir, f"{key}.bin")
                    print(f"Saving {key} to {output_path} ({one_hot_labels.shape})")
                    one_hot_labels.tofile(output_path)
        except gzip.BadGzipFile:
            print(f"Error: {filepath} is not a valid gzip file.")
            return
            
    print("\nConversion complete!")

if __name__ == "__main__":
    convert_mnist_local()