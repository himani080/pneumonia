# Pneumonia Classification using Chest X-rays

## 📌 Overview

This project uses Convolutional Neural Networks (CNNs) with hyperparameter tuning to classify chest X-ray images into two categories: **Normal** and **Pneumonia**. It is implemented using **TensorFlow** and **Keras Tuner** on **Google Colab**.

Pneumonia is a potentially fatal lung infection. Early and accurate detection using deep learning can significantly aid diagnosis and treatment, especially in resource-limited settings.

---

## 🧠 Objectives

- Understand and explore the Chest X-ray dataset.
- Study CNN architecture and the impact of hyperparameters.
- Implement and train a CNN model with hyperparameter tuning using Keras Tuner.
- Evaluate model performance on test data.

---

## 📂 Dataset

- **Source:** [Kaggle - Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Total Images:** ~5,863
- **Classes:** 
  - Normal (no pneumonia)
  - Pneumonia (includes bacterial and viral cases)

## ⚙️ Technologies Used

- Python
- TensorFlow / Keras
- Keras Tuner
- Google Colab
- Kaggle API

---

## 🔍 Hyperparameters Tuned

- Number of filters (Conv layers)
- Dropout rate
- Number of Dense layer units
- Optimizer (Adam, RMSProp, SGD)

---

## 🚀 Installation & Setup

1. Clone this repository or open it in Google Colab.

2. Upload your `kaggle.json` token.

3. Run the following commands in Colab:
   ```python
   !pip install kaggle -qq
   !mkdir -p ~/.kaggle
   !cp kaggle.json ~/.kaggle/
   !chmod 600 ~/.kaggle/kaggle.json
   !kaggle datasets download -d paultimothymooney/chest-xray-pneumonia --unzip

 Training and Evaluation

    Data is preprocessed using ImageDataGenerator with augmentation.

    Model is trained on 150x150 images with a tuned CNN architecture.

    Hyperparameter tuning is done using RandomSearch from Keras Tuner.

    Final model is evaluated on test data for accuracy and performance.

📈 Results

    Achieved high accuracy on validation and test sets.

    Effective generalization due to data augmentation and tuning.

    Robust model for binary classification using medical images.


📚 References

    Keras Documentation

    TensorFlow Tutorials

    Kaggle Dataset

💡 Future Work

    Integrate Transfer Learning using pretrained models like ResNet or VGG.

    Deploy model using a Flask or FastAPI server for web inference.

    Add Grad-CAM visualizations to highlight infected regions in X-rays.
