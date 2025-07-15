# PotatoFCN: U-Net Segmentation with oneDNN and OpenMP

**PotatoFCN** is a high-performance C implementation of the U-Net architecture for semantic segmentation on the COCO dataset. It leverages Intel oneDNN (v2.8.0) primitives and OpenMP multithreading to accelerate convolution, deconvolution, and pooling operations.

## 🚀 Features

- **oneDNN Acceleration**: Optimized `conv2d`, `deconv2d`, `maxpool`, and their gradients via oneDNN primitives.
- **OpenMP Multithreading**: Parallelized training loop for multi-core CPUs.
- **Adam Optimizer**: Bias-corrected 1st/2nd moment estimates for stable convergence.
- **Gradient Clipping**: Global norm clipping to prevent exploding gradients.
- **LR Scheduler**: Linear warmup for 500 steps, then cosine decay.
- **Checkpointing**: Save and load model parameters to/from `unet_model.bin`.
- **Batch Training**: Configurable batch size with real-time progress bar showing loss, learning rate, and elapsed time.

## ⚙️ Prerequisites

- **oneDNN v2.8.0** (Intel MKL-DNN)
- **OpenMP** support in your compiler
- **C Compiler**: GCC/Clang (Linux/macOS) or MSVC (Windows)

## 🛠️ Building

### Using CMake
```bash
mkdir build && cd build
cmake ..
cmake --build .
```

### Manual GCC Example
```bash
gcc -O3 -march=native -fopenmp src/*.c -Iinclude -ldnnl -o PotatoFCN
```

## 🎯 Usage

1. Place your preprocessed data in the `preprocessed_data/` folder:
   - `info.txt` containing three integers: `num_samples`, `height`, `width`.
   - `coco_images.bin` and `coco_masks.bin` as binary float arrays.
2. Run the executable:
   ```bash
   ./PotatoFCN
   ```
3. Monitor the training progress. Checkpoints are saved to `unet_model.bin` after each epoch.

## ⚙️ Hyperparameters

- **Epochs**: 5
- **Batch Size**: 8
- **Base Learning Rate**: 1e-4
- **Warmup Steps**: 500
- **Gradient Clip Norm**: 1.0

## 📈 Evaluation Metrics

- **Normalized MSE Loss**: aim for `< 0.01` on normalized masks.
- **Dice Coefficient**: use external script/tool, aim for `> 0.7`.

## 📄 License

This project is released under the **MIT License**. Feel free to use and modify it for your research or applications.
