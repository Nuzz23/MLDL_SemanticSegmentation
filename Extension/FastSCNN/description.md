# FastSCNN Model Overview

**FastSCNN** is a lightweight, real-time semantic segmentation network optimized for high-resolution images, making it ideal for mobile and embedded devices. Its architecture follows an encoder-decoder style, but is specifically designed for speed and low memory usage.

---

## Architecture Components

### 1. Learning to Downsample
- **Purpose:** Rapidly reduces input spatial resolution while increasing channel depth.
- **Composition:**  
    - One standard convolution layer  
    - Two depthwise separable convolutions  
- **Output:** Feature map at 1/8 the input resolution

### 2. Global Feature Extractor
- **Purpose:** Extracts high-level semantic features from the downsampled input.
- **Composition:**  
    - Three stages of Linear Bottleneck blocks (as in MobileNetV2)  
    - Pyramid Pooling Module (PPM) for multi-scale context aggregation  
- **Output:** Deep semantic features

### 3. Feature Fusion Module
- **Purpose:** Merges high-resolution spatial features with low-resolution semantic features.
- **Composition:**  
    - Upsampling of low-resolution features  
    - Depthwise and pointwise convolutions on both streams  
    - Addition of processed streams, followed by ReLU activation

### 4. Classifier
- **Purpose:** Generates the final segmentation map.
- **Composition:**  
    - Two depthwise separable convolutions  
    - Dropout for regularization  
    - Final 1x1 convolution for class logits

### 5. Auxiliary Output (Optional)
- **Purpose:** Provides auxiliary loss during training to aid gradient flow.
- **Composition:**  
    - Small convolutional head attached to the high-resolution feature stream

---

## Forward Pass

1. Input passes through the **Learning to Downsample** module.
2. Output is processed by the **Global Feature Extractor**.
3. High-resolution and low-resolution features are fused in the **Feature Fusion Module**.
4. Fused features are passed through the **Classifier** to produce segmentation logits.
5. *(Optional)* Auxiliary output is generated from high-resolution features.
6. All outputs are upsampled to the original input size.

---

## Output

- **Inference:** Returns the main segmentation map (class logits per pixel).
- **Training (if `aux=True`):** Returns a tuple with the main output and auxiliary output.

---

## Key Features

- Real-time performance on high-resolution images
- Efficient: Utilizes depthwise separable convolutions and lightweight bottleneck blocks
- Multi-scale context aggregation via Pyramid Pooling Module
- Optional auxiliary supervision for improved training convergence

