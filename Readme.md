🌍 Land Type Classification using Sentinel-2 Satellite Images

📌 Project Overview

This project focuses on building a machine learning model to classify different land cover types in Egypt using Sentinel-2 satellite imagery. The classification includes categories such as:

Agricultural Land

Water Bodies

Urban Areas

Deserts

Roads

Tree Cover

The ultimate goal is to:

Develop an accurate and robust land classification system.

Deploy the trained model through a simple web application (e.g., Streamlit) where users can upload satellite images and receive predictions.

📂 Dataset

We are using the EuroSAT RGB Dataset, which contains 27,000 labeled and geo-referenced Sentinel-2 satellite images across 10 classes.

Source: EuroSAT Dataset

Structure after extraction:

EuroSAT/
    2750/
        AnnualCrop/
        HerbaceousVegetation/
        Highway/
        Industrial/
        Pasture/
        PermanentCrop/
        SeaLake/
        Forest/
        Residential/
        River/


Each sub-folder contains images (2000–3000 per class) in RGB format (64×64 px).

🛠️ Work Completed So Far
✅ Step 1: Dataset Collection

Downloaded and extracted the EuroSAT RGB dataset.

Verified that each class folder contains thousands of images.

✅ Step 2: Data Cleaning & Preparation

Handled potential duplicates and checked for corrupted files.

Ensured dataset is in a consistent format for training.

Organized data into the following structure:

dataset/
    train/
        AnnualCrop/
        Forest/
        ...
    val/
        AnnualCrop/
        Forest/
        ...
    test/
        AnnualCrop/
        Forest/
        ...


Split dataset into:

70% Training

15% Validation

15% Testing

✅ Step 3: Dataset Summary (Metadata)

Generated a dataset_summary.json file that documents:

Number of images per class.

Distribution across train/val/test splits.

This ensures reproducibility and provides a quick overview of dataset balance.

📊 Next Steps

Perform Exploratory Data Analysis (EDA):

Class distribution visualization.

Sample image visualization.

Apply data augmentation techniques (e.g., rotation, flipping, random cropping) to improve generalization.

Begin model training using:

Baseline CNN.

Transfer learning with models such as ResNet50 or EfficientNet.

📂 Project Structure (Current)
Land-Classification-Project/
│
├── dataset/  
│   ├── train/  
│   ├── val/  
│   ├── test/  
│
├── dataset_summary.json  
├── README.md  
└── requirements.txt (to be added later)

📑 Deliverables (Completed So Far)

✅ Dataset collected and prepared.

✅ Data cleaning performed (duplicates/outliers checked).

✅ Dataset split into train/val/test.

✅ Metadata summary created (dataset_summary.json).

✅ Documentation updated in README.md.