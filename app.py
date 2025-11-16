import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

st.set_page_config(page_title="Forest Fire Detection System", page_icon="🔥")

# ---------------------------------------------------------
# Load Model
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 2)
    )
    model.load_state_dict(torch.load("/Users/anubhavverma/Documents/Fire_classification/fire_classifier_resnet50.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# ---------------------------------------------------------
# Prediction Function
# ---------------------------------------------------------
def predict(img):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, pred = torch.max(outputs, 1)

    return ["Fire", "Non_Fire"][pred.item()]

# ---------------------------------------------------------
# UI
# ---------------------------------------------------------

st.title("🔥 Forest Fire Detection System")
st.write("Upload an image to check for **Fire** or **Non_Fire**.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", width=450)

    with st.spinner("Analyzing Image..."):
        result = predict(img)

    st.subheader(f"Prediction: **{result}**")

    # ----------------------------------------
    # Action-based Suggestion System
    # ----------------------------------------
    if result == "Fire":
        st.error("⚠️ Fire Detected!")
        st.write("""
### Recommended Actions:
- Immediately alert forest authorities or emergency services.
- Evacuate the area if you are on-site.
- Avoid dry grass, leaves, or flammable materials nearby.
- If you're integrating with environmental prediction:  
  Combine this result with high-risk conditions such as **high temperature**, **low humidity**, **low rainfall**, **high wind speed**, etc.
        """)

    else:
        st.success("✔ No visible fire detected in the image.")
        st.write("""
### Suggested Steps:
- Continue monitoring conditions if your environmental model predicts high fire probability.
- If the area is prone to wildfire:  
  - Maintain safe distance  
  - Avoid activities that may ignite fire  
  - Report suspicious smoke if seen  
- Combine this output with environmental parameters for better risk assessment.
        """)

st.markdown("---")
st.caption("Model powered by PyTorch • UI built using Streamlit")
