# Explainable Deep Learning-Based Medical Image Analysis for Disease Detection and Visualization

An AI system that detects pneumonia from chest X-ray images using a deep learning model (ResNet50) and explains its predictions with **Grad-CAM heatmaps**. The results are shown in a simple Streamlit web interface.

> **Disclaimer:** This project is for educational and research purposes only. It is a decision-support tool, not a diagnostic device. Final diagnosis must always be made by a qualified medical professional.

---

## Abstract

Artificial Intelligence is playing an important role in the healthcare industry by helping doctors detect diseases more quickly and efficiently. This project develops a system that identifies diseases from medical images such as X-rays and MRI scans using Artificial Intelligence and Computer Vision.

The system uses Convolutional Neural Networks (CNN) to analyse medical images and identify possible signs of disease. A key feature is **Explainable AI (XAI)**: using Grad-CAM, the system generates heatmaps that highlight the regions of the image that influenced the prediction. Along with the predicted result, the interface displays a confidence score and a visual explanation, making the output easier to understand and helping doctors examine the areas the model considers important.

The aim is not only to detect disease but also to make AI predictions more transparent and understandable.

---

## Features

- Pneumonia detection from chest X-rays (Normal vs Pneumonia)
- Transfer learning with a pretrained ResNet50
- Grad-CAM heatmap visualization overlaid on the original image
- Prediction label with confidence score
- Simple, user-friendly Streamlit interface

---

## Demo

Add your screenshots here (create an `images/` folder in the repo):

| Upload | Prediction + Grad-CAM |
|---|---|
| ![Upload](images/upload.png) | ![Result](images/result.png) |

---

## Project Structure

```
explainable-medical-image-analysis/
├── train.py            # Model training script
├── gradcam.py          # Grad-CAM implementation
├── app.py              # Streamlit web app
├── split_val.py        # Creates a larger validation set from training data
├── requirements.txt    # Python dependencies
├── .gitignore
└── README.md
```

The dataset (`data/`), virtual environment (`venv/`), and trained weights (`best_model.pth`) are not stored in this repository.

---

## Tech Stack

- Python 3.9 - 3.11
- PyTorch and torchvision
- OpenCV
- scikit-learn
- Streamlit
- NumPy, Pillow, Matplotlib

---

## Dataset

**Chest X-Ray Images (Pneumonia)** by Paul Mooney on Kaggle:
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Expected folder layout after download:

```
data/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

The original `val` folder is very small (16 images), so `split_val.py` moves 15% of the training images into `val` for more reliable validation. Run it once:

```bash
python split_val.py
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/explainable-medical-image-analysis.git
cd explainable-medical-image-analysis

# 2. Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate
# Mac / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Training

A GPU is strongly recommended. Training ResNet50 on a CPU is very slow, so the model in this project was trained on Google Colab (free T4 GPU).

```bash
python train.py
```

This saves the best model as `best_model.pth`.

Training settings: ResNet50 (ImageNet pretrained), Adam optimizer, learning rate 1e-4, image size 224x224, data augmentation (random flip and rotation).

---

## Pretrained Model

The trained weights are not included in this repository because of file size.

Download `best_model.pth` here: **[add your Google Drive link]**

Place it in the project root folder (next to `app.py`).

---

## Running the App

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser, upload a chest X-ray image, and view the prediction, confidence score, and Grad-CAM heatmap.

---

## How It Works

```
Upload Image -> Preprocessing -> ResNet50 -> Prediction + Confidence
                                     |
                                     v
                                 Grad-CAM -> Heatmap overlay -> Streamlit UI
```

1. The image is resized to 224x224 and normalized.
2. ResNet50 predicts the class (Normal or Pneumonia) and a confidence score.
3. Grad-CAM uses the gradients of the predicted class with respect to the last convolutional layer (`layer4[-1]`) to build a heatmap.
4. The heatmap is overlaid on the original X-ray, highlighting the regions that most influenced the prediction.

---

## Results

Fill in these values from your test set evaluation:

| Metric | Value |
|---|---|
| Accuracy | _xx.x%_ |
| Precision | _xx.x%_ |
| Recall (Sensitivity) | _xx.x%_ |
| F1-score | _xx.x%_ |
| AUC | _0.xx_ |

Add your confusion matrix and training curves in `images/`.

---

## Limitations

- Trained on a single public dataset, so results may not generalize to other hospitals or imaging devices.
- Not clinically validated.
- Grad-CAM highlights are an aid to interpretation and do not prove the model is reasoning the way a clinician would.
- Currently supports only binary classification (Normal vs Pneumonia).

---

## Future Work

- Add brain tumor detection from MRI scans
- Multi-disease classification
- Compare more models (DenseNet121, EfficientNet)
- Add other XAI methods (Grad-CAM++, LIME, SHAP)
- Downloadable PDF report of each prediction

---

## Acknowledgements

- Dataset: Paul Mooney, Kaggle (originally from Kermany et al.)
- Grad-CAM: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
- PyTorch and Streamlit communities

---

## Author

Md Arshed Patel
GitHub: [@arxed](https://github.com/arxed)

