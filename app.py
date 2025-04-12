import streamlit as st
import os
import tempfile
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import gdown

# Constants
MODEL_PATH = "handwriting_personality_model_final.h5"
GDRIVE_FILE_ID = "1-1u1rUYd8CC5HZrIu6Js8v2hhZvP_-H1"  
TARGET_IMAGE_SIZE = (224, 224)  # Adjust based on your model input size

# Descriptions and indicators
trait_descriptions = {
    "Openness": "People high in Openness are curious, imaginative, and open to new experiences...",
    "Conscientiousness": "Highly conscientious individuals are organized, detail-oriented, and responsible...",
    "Extraversion": "Extraverts are outgoing, enthusiastic, and sociable...",
    "Agreeableness": "Agreeable individuals are compassionate, cooperative, and trusting...",
    "Neuroticism": "People with high neuroticism tend to experience emotional instability, anxiety, and mood swings..."
}

handwriting_indicators = {
    "Openness": "Wide spacing, artistic flourishes, inconsistent baselines...",
    "Conscientiousness": "Consistent slant, even spacing, clear characters...",
    "Extraversion": "Large letter size, upward slant, fast speed...",
    "Agreeableness": "Rounded letters, connected strokes, slight right slant...",
    "Neuroticism": "Variable slant, sharp angles, irregular spacing..."
}

# Load model from GDrive (cached)
@st.cache_resource
def load_model_cached():
    if not os.path.exists(MODEL_PATH):
        gdown.download(f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}", MODEL_PATH, quiet=False)
    return load_model(MODEL_PATH)

model = load_model_cached()

# Image preprocessing
def preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).convert("RGB").resize(target_size)
    img_array = np.array(img) / 255.0  # Normalize
    return np.expand_dims(img_array, axis=0)

# Model inference and formatting results
def analyze_handwriting_personality(image_path, model):
    preprocessed_img = preprocess_image(image_path, target_size=TARGET_IMAGE_SIZE)
    predictions = model.predict(preprocessed_img)[0]  # Output shape: (5,)

    trait_labels = list(trait_descriptions.keys())
    results = {}

    for i, trait in enumerate(trait_labels):
        confidence = float(predictions[i])
        results[trait] = {
            "confidence": confidence,
            "is_primary": False
        }

    # Highlight the most confident trait
    primary_trait = max(results, key=lambda k: results[k]["confidence"])
    results[primary_trait]["is_primary"] = True

    return primary_trait, results[primary_trait]["confidence"], results

# Streamlit UI
st.set_page_config(page_title="Handwriting Personality Analyzer", layout="centered")
st.title("🖋️ Handwriting Personality Analyzer")
st.write("Upload handwriting to discover personality traits using deep learning.")

uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Analyze Handwriting"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            image.save(tmp.name)
            image_path = tmp.name

            predicted_trait, confidence, results = analyze_handwriting_personality(image_path, model)

            # Show results
            st.subheader("🧠 Primary Personality Trait")
            st.success(f"{predicted_trait} (Confidence: {confidence:.2f})")

            st.markdown("### 📘 Description")
            st.write(trait_descriptions[predicted_trait])
            st.markdown("### ✍️ Handwriting Indicators")
            st.write(handwriting_indicators[predicted_trait])

            # Bar chart
            st.markdown("### 📊 Trait Confidence Chart")
            fig, ax = plt.subplots(figsize=(10, 6))
            trait_names = list(results.keys())
            confidences = [results[t]["confidence"] for t in trait_names]
            colors = ['green' if results[t]["is_primary"] else 'skyblue' for t in trait_names]

            bars = ax.bar(trait_names, confidences, color=colors, edgecolor='black')
            ax.set_ylim(0, 1)
            ax.set_ylabel("Confidence", fontsize=12)
            ax.set_title("Predicted Personality Traits from Handwriting", fontsize=14)
            ax.grid(axis='y', linestyle='--', alpha=0.7)

            for bar, conf in zip(bars, confidences):
                ax.text(bar.get_x() + bar.get_width() / 2, conf + 0.02, f'{conf:.2f}', ha='center', fontsize=10, fontweight='bold')

            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            st.pyplot(fig)

            st.markdown("---")
            st.caption("⚠️ This tool is for educational purposes only and not scientifically validated.")

            os.unlink(image_path)