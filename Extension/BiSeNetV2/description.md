# BiSeNet V2
BiSeNet V2 is an efficient and effective model for semantic segmentation, improving upon its predecessor, BiSeNet V1.

## Description

BiSeNet V2 retains a two-branch architecture but introduces several enhancements to boost performance and reduce computational cost:

- **Improved Spatial Path:** Captures high-resolution features more effectively, preserving spatial information crucial for segmentation.
- **Enhanced Context Path:** Optimized to better capture contextual information, enabling more informed segmentation decisions.
- **Lightweight Design:** Employs a more efficient structure, making the model faster and more resource-friendly while maintaining high accuracy.

## Backbone
BiSeNet V2 introduces a custom, lightweight backbone specifically designed for efficiency and real-time performance. It does not rely on external pre-trained networks like ResNet-18. Instead, its backbone is built from scratch to balance speed and accuracy, making it suitable for deployment on resource-constrained devices.

## Key Differences Compared to BiSeNet V1

BiSeNet V2 achieves higher FPS (Frames Per Second) and better segmentation accuracy due to:

- **Optimized Spatial Path:** More efficient, capturing spatial information with fewer computations.
- **Lightweight Context Path:** Reduces computational load while maintaining global context.
- **Parallel Design:** Processes spatial and context information simultaneously, speeding up inference.
- **Fewer Parameters:** Reduces model size and operations, increasing speed without significant accuracy loss.

These improvements enable real-time performance, making BiSeNet V2 suitable for applications requiring fast segmentation.

Additionally, BiSeNet V2 achieves higher mIoU (mean Intersection over Union) through:

- **Better Feature Extraction:** Enhanced backbone and spatial path for more effective feature extraction.
- **Improved Contextual Understanding:** Optimized context path for capturing relevant context.
- **Advanced Training Techniques:** Utilizes refined training and data augmentation strategies for better generalization.

## Summary of Differences: BiSeNet vs. BiSeNet V2

| Aspect                | BiSeNet (V1)                                                                 | BiSeNet V2                                                      |
|-----------------------|------------------------------------------------------------------------------|-----------------------------------------------------------------|
| **Architecture**      | Two paths: Spatial (ConvBlocks) and Context (ResNet-18/101); uses attention modules and feature fusion | Detail Branch (spatial details), Semantic Branch (context); BGALayer for fusion |
| **Backbone**          | Standard ResNet-18/101                                                       | Custom lightweight backbone (no external ResNet)                |
| **Fusion Mechanism**  | Attention Refinement Modules, Feature Fusion Module                          | BGALayer (Bilateral Guided Aggregation)                         |
| **Auxiliary Outputs** | Two from context path                                                        | Four from semantic branch stages                                |
| **Implementation**    | Modular, backbone-agnostic, more complex                                     | Self-contained, efficient, all modules defined in one file      |
| **Usage**             | Suitable for strong backbone scenarios                                       | Optimized for efficiency and speed, ideal for edge devices      |

BiSeNet V2 is designed for real-time, efficient semantic segmentation, especially on resource-constrained devices.
