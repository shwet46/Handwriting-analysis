import streamlit as st
import os
import tempfile
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.models import load_model
import gdown

# Download the model from Google Drive if it doesn't exist
MODEL_PATH = "handwriting_personality_model_final.h5"
GDRIVE_FILE_ID = "YOUR_FILE_ID"  # replace this

@st.cache_resource
def load_model_cached():
    if not os.path.exists(MODEL_PATH):
        gdown.download(f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}", MODEL_PATH, quiet=False)
    return load_model(MODEL_PATH)

model = load_model_cached()

# Dummy dictionaries (replace with real content)
trait_descriptions = {
    "Openness": "Curious, creative, and open to new experiences and ideas.",
    "Conscientiousness": "Organized, disciplined, responsible, and detail-oriented.",
    "Extraversion": "Outgoing, energetic, sociable, and assertive.",
    "Agreeableness": "Trustworthy, helpful, cooperative, and empathetic.",
    "Neuroticism": "Emotionally reactive and prone to stress."
}

handwriting_indicators = {
    "Openness": "Large loops, varying baseline.",
    "Conscientiousness": "Neat, consistent slant and spacing.",
    "Extraversion": "Large letters, upward slant.",
    "Agreeableness": "Rounded letters, light pressure.",
    "Neuroticism": "Erratic slant, heavy pressure."
}

# Dummy analyzer (replace with actual logic)
def analyze_handwriting_personality(image_path, model):
    results = {
        "Openness": {"confidence": 0.87, "is_primary": True},
        "Conscientiousness": {"confidence": 0.65, "is_primary": False},
        "Extraversion": {"confidence": 0.44, "is_primary": False},
        "Agreeableness": {"confidence": 0.39, "is_primary": False},
        "Neuroticism": {"confidence": 0.22, "is_primary": False},
    }
    predicted_trait = max(results, key=lambda k: results[k]["confidence"])
    confidence = results[predicted_trait]["confidence"]
    return predicted_trait, confidence, results

# UI
st.title("🖋️ Handwriting Personality Analyzer")
st.write("Upload handwriting to discover personality traits using deep learning.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Analyze Handwriting"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            image.save(tmp.name)
            image_path = tmp.name

            predicted_trait, confidence, results = analyze_handwriting_personality(image_path, model)

            st.subheader("Primary Personality Trait")
            st.success(f"{predicted_trait} (Confidence: {confidence:.2f})")

            st.markdown("### Description")
            st.write(trait_descriptions[predicted_trait])
            st.markdown("### Handwriting Indicators")
            st.write(handwriting_indicators[predicted_trait])

            # Plot chart
            st.markdown("### Trait Confidence Chart")
            fig, ax = plt.subplots()
            trait_names = list(results.keys())
            confidences = [results[t]["confidence"] for t in trait_names]
            colors = ['green' if results[t]["is_primary"] else 'skyblue' for t in trait_names]
            bars = ax.bar(trait_names, confidences, color=colors)
            ax.set_ylim(0, 1)
            ax.set_ylabel("Confidence")
            ax.set_title("Personality Traits")

            for bar, conf in zip(bars, confidences):
                ax.text(bar.get_x() + bar.get_width() / 2, conf + 0.02, f'{conf:.2f}', ha='center', fontsize=9)

            st.pyplot(fig)

            st.markdown("---")
            st.caption("⚠️ This tool is for educational purposes only and not scientifically validated.")

            os.unlink(image_path)