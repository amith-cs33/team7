
# 🫁 Lung Cancer Detection Using Deep Learning (CNN + MATLAB)

![License](https://img.shields.io/badge/license-MIT-green)
![Language](https://img.shields.io/badge/MATLAB-CNN-blue)

## 📌 Project Overview

Lung cancer is the **leading cause of cancer-related deaths** globally, primarily due to **late detection**. This project presents a deep learning-based solution using **Convolutional Neural Networks (CNNs)** integrated with a **MATLAB GUI** for detecting and classifying lung tumors in **CT scan images**.

🔍 The model classifies images into:
- ✅ **Normal**
- ⚠️ **Benign (non-cancerous)**
- ❌ **Malignant (cancerous)**

The GUI also provides **tumor classification**, and **real-time detection**, enhancing early diagnosis and aiding healthcare professionals.

---

## 🚀 Key Features

- 🔎 **Automatic tumor classification** from CT scans
- 🧠 Built using a custom **CNN architecture**
- 🖥️ **User-friendly MATLAB GUI**
- 📊 **Confusion matrix and performance metrics**
- 🧪 **Data augmentation** for improved model generalization
- ⚡ **GPU-accelerated training** using MATLAB's Deep Learning Toolbox

---

## 📷 Screenshots

| Input Image | Segmentation | Classification |
|-------------|--------------|----------------|
| ![Input](screenshots/input.jpg) | ![Segmented](screenshots/segmentation.jpg) | ![Output](screenshots/output.jpg) |

---

## 📁 Project Structure

```
LungCancerDetection/
├── Lung_cancer_dataset/
│   ├── Train/
│   └── Test/
├── lung_cancer_model.mat
├── main.m           # Model training script
├── LungCancerGUI.m  # GUI implementation
├── README.md
└── screenshots/
```

---

## ⚙️ Installation & Setup

1. **Requirements:**
   - MATLAB R2021b or higher
   - Deep Learning Toolbox
   - Image Processing Toolbox
   - GPU support (optional but recommended)

2. **Clone the repo**
   ```bash
   git clone https://github.com/amith-cs33/team7.git
   cd LungCancerDetection
   ```

3. **Run the training script**
   ```matlab
   main.m
   ```

4. **Launch the GUI**
   ```matlab
   LungCancerGUI
   ```

---

## 📊 Model Architecture

Custom CNN Model:
- Input size: `512x512x3`
- 4 Convolutional blocks (32 → 64 → 128 → 256 filters)
- Batch Normalization + ReLU + MaxPooling
- Fully Connected layers (256 → 128 → 3)
- Dropout for regularization
- Softmax + Classification Layer

Training:
- Optimizer: Adam
- Learning rate: 0.0001 with piecewise decay
- Epochs: 100
- Batch Size: 32

---

## 💡 How It Works

1. Upload CT scan via the GUI.
2. Image is preprocessed (grayscale → noise filtering → edge detection).
3. CNN model classifies the image (Normal / Benign / Malignant).
4. The GUI displays the result and highlights tumor area with segmentation.
5. Outputs tumor size, location, and classification.

---

## 📚 Dataset Sources

- [LIDC-IDRI](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI)
- [LUNA16](https://luna16.grand-challenge.org/)
- NIH Chest X-Ray Dataset (optional)

---

## 🧠 Contributors

- **Amith R** - ENG21CS0033
- **D Akshitha** - ENG21CS0102
- **Suhagh A N** - ENG21CS0423
- **Vinutha V** - ENG21CS0476

Supervised by **Dr. Pramod Kumar Naik** – Chairperson, Dept. of AI & Robotics, DSU

---

## 📌 Future Enhancements

- Integration with cloud-based telemedicine platforms
- PET/CT fusion support
- Mobile application deployment
- Explainable AI features (Grad-CAM, Heatmaps)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙌 Acknowledgements

- Dayananda Sagar University, School of Engineering  
- MATLAB Deep Learning Toolbox Team  
- Open-source medical imaging communities
