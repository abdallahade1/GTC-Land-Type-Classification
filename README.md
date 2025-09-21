#  Land Cover Classification using Deep Learning

A comprehensive deep learning project for classifying land cover types from satellite imagery using NWPU-RESISC45 dataset.

##  Project Overview

This project addresses the critical need for accurate land type classification, which is essential for applications like:
-  Agriculture monitoring
-  Urban planning
-  Water resource management
-  Environmental studies

##  Project Goals

Develop a robust land classification system that identifies 45 different land cover categories including:
- Agricultural land, water bodies, urban areas
- Deserts, roads, tree cover
- Airports, bridges, forests, and more

##  Key Features

- **Multi-Model Architecture**: ResNet50 and EfficientNetB0 implementations
- **Advanced Fine-tuning**: Transfer learning with optimized hyperparameters
- **Comprehensive EDA**: Detailed exploratory data analysis and visualizations
- **Performance Analysis**: Extensive evaluation metrics and model comparison
- **Production Ready**: Trained models ready for deployment

##  Dataset

**NWPU-RESISC45 Dataset**
- **45 land cover classes**
- **~31,500 satellite images** (700 per class)
- **Image resolution**: Variable (resized to 224224)
- **Source**: Northwestern Polytechnical University

##  Technical Implementation

### Models Implemented
1. **ResNet50**
   - Pre-trained on ImageNet
   - Custom classifier head
   - Transfer learning approach

2. **EfficientNetB0**
   - Pre-trained EfficientNet-B0 backbone
   - Advanced fine-tuning with differential learning rates
   - Cosine annealing scheduler
   - Early stopping mechanism

##  Performance Results

### Model Comparison
| Model | Validation Accuracy | Precision | Recall | F1-Score |
|-------|-------------------|-----------|--------|----------|
| ResNet50 | 81.10% | 0.812 | 0.811 | 0.811 |
| EfficientNetB0 | **83.45%** | **0.835** | **0.834** | **0.834** |

##  Installation & Setup

### Prerequisites
`ash
Python 3.8+
PyTorch 1.12+
CUDA 11.6+ (for GPU acceleration)
`

### Required Packages
`ash
pip install torch torchvision torchaudio
pip install efficientnet-pytorch
pip install opencv-python
pip install scikit-learn
pip install matplotlib seaborn
pip install pandas numpy
pip install tqdm
pip install pillow
`

##  Usage

### Training Models
`python
# Run the complete training pipeline
jupyter notebook GTC.ipynb
`

##  Project Structure

`
land-cover-classification/
 README.md
 GTC.ipynb                 # Main notebook with complete pipeline
 NWPU-RESISC45/            # Dataset directory
    airplane/
    airport/
    beach/
    ... (45 classes)
`

##  References

- [NWPU-RESISC45 Dataset](http://www.escience.cn/people/gongcheng/NWPU-RESISC45.html)
- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

##  Authors

- **Your Name** - *Initial work*

##  Contact

- **Email**: your.email@example.com
- **GitHub**: [Your GitHub](https://github.com/yourusername)

---

 **Star this repository if you found it helpful!**
