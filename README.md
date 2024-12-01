# Conditional Generative Adversarial Network (CGAN) for Image Super-Resolution

This repository contains a TensorFlow-based implementation of a Conditional Generative Adversarial Network (CGAN) designed for image super-resolution. The goal is to upscale low-resolution (LR) images to high-resolution (HR) images using adversarial training.

---

## Features

- **Custom Dataset Loading**: Load HR images from a directory, normalize them, and dynamically generate LR images.
- **Generator Model**: Creates high-resolution images with enhanced details using residual blocks.
- **Discriminator Model**: Differentiates between real HR images and generated ones.
- **Custom Training Loop**: Implements adversarial training using a custom CGAN class.
- **Callbacks**: Save generated images during training for visualization.

---

## Requirements

- Python 3.x
- TensorFlow 2.x
- OpenCV
- NumPy
- Matplotlib

Install dependencies using:
```bash
pip install tensorflow opencv-python numpy matplotlib
