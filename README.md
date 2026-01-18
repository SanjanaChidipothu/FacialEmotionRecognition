# Facial Emotion Recognition using Deep Learning

## 📌 Project Overview
Facial Emotion Recognition (FER) is the task of identifying human emotions from facial images.  
This project leverages **deep learning techniques** to classify facial expressions into seven emotion categories using the **FER2013 dataset**.

The goal of this project is to build an accurate and efficient emotion recognition model suitable for **real-time and mobile applications**, enabling systems to adapt behavior based on a user’s emotional state.

---

## 🎯 Objectives
- Perform facial emotion classification using deep learning
- Compare multiple CNN-based architectures
- Improve model performance through preprocessing and class balancing
- Deploy the best-performing model for real-time emotion detection

---

## 📊 Dataset
- **Dataset Name:** FER2013  
- **Source:** Kaggle  
- **Images:** 35,887 grayscale facial images  
- **Image Size:** 48 × 48 pixels  
- **Emotion Classes (7):**
  - Angry
  - Disgust
  - Fear
  - Happy
  - Sad
  - Surprise
  - Neutral  

⚠️ The dataset is **imbalanced**, which required additional preprocessing strategies.

---

## ⚙️ Data Preprocessing
- **Label Encoding:** Converted emotion labels into numerical values
- **Normalization:** Scaled pixel values to improve model convergence
- **Class Balancing:**  
  - Applied both **oversampling** and **undersampling** techniques
- **Data Augmentation:**  
  - Used `ImageDataGenerator` to improve generalization

---

## 🧠 Methodology
Multiple deep learning architectures were explored and evaluated:

### Models Implemented
- **Custom Conv2D CNN**
- **VGG16**
- **InceptionV3**

Each model was trained and evaluated independently on the FER2013 dataset.

### Final Model Selection
- The **Custom Conv2D model** achieved the **highest accuracy** and was selected for deployment.

---

## 🏗️ Model Architecture (Conv2D)
- Convolutional Layers (Conv2D)
- ELU Activation Functions
- Batch Normalization
- Max Pooling Layers
- Dropout (to prevent overfitting)
- Fully Connected Dense Layers

### Training Configuration
- **Loss Function:** Categorical Cross-Entropy  
- **Optimizers:** Adam / Nadam  
- **Evaluation Metric:** Accuracy  
- **Callbacks:**
  - EarlyStopping
  - ReduceLROnPlateau

---

## 📈 Results
- **Best Test Accuracy:** **71%**
- The Conv2D model outperformed VGG16 and InceptionV3 on the FER2013 dataset
- Achieved stable performance despite dataset imbalance

---

## 🎥 Real-Time Implementation
- Implemented **real-time facial emotion detection** using **OpenCV**
- Captured live video feed
- Detected faces and predicted emotions frame-by-frame
- Demonstrated feasibility for real-world applications

---

## 🛠️ Tools & Technologies
- **Python**
- **TensorFlow / Keras**
- **OpenCV**
- **NumPy**
- **Matplotlib**
- **ImageDataGenerator**
- **Deep CNNs (Conv2D)**
- **VGG16**
- **InceptionV3**

---

## 📁 Project Structure
