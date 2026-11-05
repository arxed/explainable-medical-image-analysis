# app.py
import streamlit as st, torch, numpy as np
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn
from gradcam import GradCAM, overlay_heatmap

classes = ["NORMAL", "PNEUMONIA"]  # match your folder order
device = "cpu"

@st.cache_resource
def load_model():
    m = models.resnet50()
    m.fc = nn.Linear(m.fc.in_features, len(classes))
    m.load_state_dict(torch.load("best_model.pth", map_location=device))
    return m.eval()

model = load_model()
cam_tool = GradCAM(model, model.layer4[-1])

tf = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

st.title("Explainable AI Medical Image Analysis")
file = st.file_uploader("Upload X-ray / MRI", type=["jpg","jpeg","png"])

if file:
    img = Image.open(file).convert("RGB")
    x = tf(img).unsqueeze(0)
    cam, idx, conf = cam_tool.generate(x)
    overlay = overlay_heatmap(cam, np.array(img.resize((224, 224))))

    c1, c2 = st.columns(2)
    c1.image(img, caption="Original", use_column_width=True)
    c2.image(overlay, caption="Grad-CAM Heatmap", use_column_width=True)
    st.success(f"Prediction: **{classes[idx]}**  |  Confidence: **{conf*100:.1f}%**")
    st.warning("This tool is for decision support only. Final diagnosis must be made by a qualified medical professional.")