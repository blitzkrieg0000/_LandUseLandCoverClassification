
# 🌍 Semantic Segmentation Framework

This repository contains a modular framework for training, evaluating, and deploying **semantic segmentation models** such as **DeepLabv3** and **U-Net**.  
It integrates dataset preprocessing, training pipelines, ONNX export, and multiple UI frontends for visualization.

---

## 🌱 Land Use / Land Cover (LULC)

Land Use / Land Cover (LULC) mapping is the process of **classifying and analyzing the Earth's surface** based on satellite or aerial imagery.  
This is essential in **environmental monitoring, urban planning, agriculture, and climate studies**.

### 🔹 Key Concepts
- **Land Cover**: Physical material on the surface (e.g., water, forest, urban areas, crops).  
- **Land Use**: Human activities or purposes associated with a land cover type (e.g., residential, industrial, agricultural).  

### 🔹 LULC Segmentation with Deep Learning
- **Semantic Segmentation**: Pixel-wise classification to label each pixel with a land cover/use class.  
- **Common Models**: DeepLabv3, U-Net, 3D U-Net.  
- **Input Data**: Satellite imagery, multi-spectral images, or raster datasets.  
- **Output**: Classified maps showing LULC classes per pixel.  

### 🔹 Applications
- Urban growth monitoring  
- Deforestation and forest cover change detection  
- Crop type mapping and precision agriculture  
- Flood monitoring and water resource management  

### 🔹 Datasets
- Public datasets: **Landsat**, **Sentinel-2**, **MODIS**, **Corine Land Cover**  
- Custom datasets created from high-resolution satellite imagery  

---

### 🔹 Integration in This Framework
- Preprocessing pipelines for satellite imagery  
- RasterLoader and Dataset modules handle LULC datasets  
- Trained DeepLabv3 / U-Net models predict LULC classes  
- Dash/Gradio UI for visualizing LULC maps interactively

---

## 📂 Project Structure

### 📊 Dataset Management
- Base classes and processors for loading raster/geospatial datasets  
- Dataset transformations and preprocessing utilities  
- Integration with **Google Earth Engine (GEE)** and **STAC catalogs**  

### 🧠 Models
- DeepLabv3 (with ResNet50 backbone)  
- U-Net and 3D U-Net implementations  
- Modular loss functions and training utilities  
- ONNX export for inference in lightweight environments  

### 🏋️ Training & Evaluation
- Training pipelines with configurable constants and core training loops  
- Sliding window inference for large images  
- Evaluation scripts and notebook examples  
- Checkpoint saving and weight management  

### 🔮 Prediction & Deployment
- Inference with PyTorch models  
- ONNX-based prediction for optimized performance  
- Sliding window prediction for handling big raster data  

### 🖥️ User Interfaces
- **Dash** interactive maps for visualization  
- **Gradio** demos for quick prototyping  
- **FastAPI** endpoints for serving models  
- Map rendering in HTML for dataset inspection  

### 🛠️ Tools
- Data storage and indexing helpers  
- Utility scripts for managing experiments  
- Integration with **Weights & Biases (wandb)** for experiment tracking  

### 📦 Weights
- Pretrained weights for DeepLabv3 and U-Net models  
- Organized storage for reproducibility
