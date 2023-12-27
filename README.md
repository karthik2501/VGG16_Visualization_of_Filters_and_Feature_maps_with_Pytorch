# VGG16 Visualizations with PyTorch

![vgg16_architecture](https://github.com/karthik2501/VGG16_Visualizations_with_Pytorch/assets/75373017/b310c216-17fd-4c43-80c8-60135ebcfb91)

Image source: [ResearchGate](https://www.researchgate.net/figure/A-sample-architecture-of-VGG-16-Image-source-42_fig3_343092954)


This repository contains Python scripts to visualize filters and feature maps from different layers of the VGG16 model using PyTorch. The VGG16 model is a widely used convolutional neural network architecture for image classification.

### Visualization Examples

#### 1. Filter Visualization
![vgg16_filters](https://github.com/karthik2501/VGG16_Visualizations_with_Pytorch/assets/75373017/6e5c7693-4c50-4159-bbe7-85d69b551f12)

The script [`visualize_filters.py`](visualize_filters.py) visualizes the filters learned by a chosen convolutional layer in the VGG16 model.

#### 2. Feature Map Visualization
![feature_maps_each_layer](https://github.com/karthik2501/VGG16_Visualizations_with_Pytorch/assets/75373017/4d278186-f67c-411c-a61d-31084c0bf21b)

The script [`visualize_feature_maps_each_layer.py`](visualize_feature_maps_each_layer.py) generates feature maps for each layer of the VGG16 model using a sample image.

#### 3. Feature Maps from Different Layers
![feature_maps_different_layers](https://github.com/karthik2501/VGG16_Visualizations_with_Pytorch/assets/75373017/716136bb-84b9-4b7e-8642-4c35b6018d10)

The script [visualize_feature_maps_different_layers](visualize_feature_maps_different_layers) visualize_feature_maps_different_layers.py demonstrates feature maps from selected layers of the VGG16 model using a sample image.

### Usage
- Clone the repository:
  ```bash
  git clone https://github.com/karthik2501/VGG16_Visualizations_with_Pytorch.git
  cd VGG16_Visualizations_with_Pytorch
  
- Run the scripts using Python:
  ```bash
  python visualize_filters.py
  python visualize_feature_maps_each_layer.py
  python visualize_feature_maps_different_layers.py

### Requirements
- Python 3.x
- PyTorch
- torchvision
- Matplotlib

### Acknowledgments
- The VGG16 model architecture is credited to the authors of the paper ["Very Deep Convolutional Networks for Large-Scale Image Recognition"](https://arxiv.org/abs/1409.1556).

Feel free to explore and modify the scripts for different layers and images!

  
