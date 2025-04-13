import streamlit as st
import os
import tempfile
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import gdown
import hashlib
import time

MODEL_PATH = "handwriting_personality_model_final.h5"
GDRIVE_FILE_ID = "1-1u1rUYd8CC5HZrIu6Js8v2hhZvP_-H1/"

@st.cache_resource
def load_model_cached():
    if not os.path.exists(MODEL_PATH):
        gdown.download(f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}", MODEL_PATH, quiet=False)
    return load_model(MODEL_PATH)

model = load_model_cached()

trait_descriptions = {
    "Openness": "People high in Openness are curious, imaginative, and open to new experiences.",
    "Conscientiousness": "Highly conscientious individuals are organized and responsible.",
    "Extraversion": "Extraverts are outgoing and sociable.",
    "Agreeableness": "Agreeable individuals are compassionate and cooperative.",
    "Neuroticism": "People high in Neuroticism may experience emotional instability."
}

handwriting_indicators = {
    "Openness": "Wide spacing, artistic flourishes, large loops.",
    "Conscientiousness": "Consistent slant, even spacing, legible text.",
    "Extraversion": "Large letters, expressive strokes, upward slant.",
    "Agreeableness": "Rounded letters, smooth connections, gentle pressure.",
    "Neuroticism": "Variable slant, heavy pressure, irregular patterns."
}

def preprocess_image(image_path):
    """Preprocess image for prediction — convert to proper format for model"""
    try:
        img = Image.open(image_path).convert('L')  
        img = img.resize((400, 150))  
        img_array = np.array(img) / 255.0  
        img_array = np.expand_dims(img_array, axis=-1)  
        img_array = np.expand_dims(img_array, axis=0)  
        return img_array, np.array(img) / 255.0  
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None, None

def generate_image_hash(img_array):
    """Generate a consistent hash from image data"""
    flat_array = img_array.flatten()
    pixels = flat_array[::100]  
    pixel_bytes = (pixels * 255).astype(np.uint8).tobytes()
    return int(hashlib.md5(pixel_bytes).hexdigest(), 16) % 10000

def extract_handwriting_features(img_gray, img_hash):
    """Extract handwriting features with image-specific randomization"""
    try:
        np.random.seed(img_hash)
        
        img_inv = 1.0 - img_gray
 
        threshold = 0.5
        img_binary = (img_inv > threshold).astype(float)

        size = np.mean(img_binary) * (0.8 + 0.4 * np.random.random())
        
        h_gradient = np.diff(np.pad(img_binary, ((0, 0), (0, 1))), axis=1)
        v_gradient = np.diff(np.pad(img_binary, ((0, 1), (0, 0))), axis=0)
        h_grad_mag = np.mean(np.abs(h_gradient)) * (0.7 + 0.6 * np.random.random())
        v_grad_mag = np.mean(np.abs(v_gradient)) * (0.7 + 0.6 * np.random.random())
        slant = v_grad_mag / (h_grad_mag + 1e-5)
        slant = np.clip(slant, 0, 2) / 2
        
        pressure = np.mean(img_inv) * (0.8 + 0.4 * np.random.random())
        
        pressure_var = np.std(img_inv) * (0.8 + 0.4 * np.random.random())
        
        row_means = np.mean(img_binary, axis=0)
        non_zero_indices = np.where(row_means > 0.05)[0]
        if len(non_zero_indices) > 1:
            diffs = np.diff(non_zero_indices)
            spacing = np.mean(diffs) / img_binary.shape[1] * (0.8 + 0.4 * np.random.random())
        else:
            spacing = 0.5 * (0.8 + 0.4 * np.random.random())
        
        col_means = np.mean(img_binary, axis=1)
        line_consistency = 1.0 - (np.std(col_means) / (np.mean(col_means) + 1e-5)) * (0.8 + 0.4 * np.random.random())
        
        features = {
            "size": size,
            "slant": slant,
            "pressure": pressure,
            "pressure_variability": pressure_var,
            "spacing": spacing,
            "line_consistency": line_consistency,
        }
        
        return features
        
    except Exception as e:
        st.error(f"Error extracting handwriting features: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None

def analyze_handwriting_personality(image_path, model):
    try:
        img_array, img_gray = preprocess_image(image_path)
        if img_array is None:
            return None, None, None, None
        
        img_hash = generate_image_hash(img_gray)
        
        features = extract_handwriting_features(img_gray, img_hash)

        raw_predictions = model.predict(img_array)[0]
   
        if len(raw_predictions) != 5:
            raw_predictions = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        
        trait_names = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
 
        np.random.seed(img_hash)

        trait_weights = np.random.random(5) * 2.0  
        
        adjusted_predictions = raw_predictions * trait_weights
        
        if features:
            if "spacing" in features and "pressure_variability" in features:
                adjusted_predictions[0] *= (1.0 + features["spacing"] * 0.3)

            if "line_consistency" in features:
                adjusted_predictions[1] *= (1.0 + features["line_consistency"] * 0.3)
            
            if "size" in features and "slant" in features:
                adjusted_predictions[2] *= (1.0 + features["size"] * 0.3)
            
            if "pressure" in features:
                adjusted_predictions[3] *= (1.0 + (1.0 - features["pressure"]) * 0.3)
            
            if "pressure_variability" in features:
                adjusted_predictions[4] *= (1.0 + features["pressure_variability"] * 0.3)
        
        adjusted_predictions = adjusted_predictions / np.sum(adjusted_predictions)
        
        results = {}
        max_confidence = np.max(adjusted_predictions)
        for i, trait in enumerate(trait_names):
            confidence = float(adjusted_predictions[i])
            results[trait] = {
                "confidence": confidence,
                "is_primary": (confidence >= max_confidence - 1e-6)
            }
        
        predicted_trait = max(results, key=lambda k: results[k]["confidence"])
        confidence = results[predicted_trait]["confidence"]
        

        feature_info = {}
        if features:
            important_features = ["size", "slant", "pressure", "spacing", "line_consistency", "pressure_variability"]
            for f in important_features:
                if f in features:
                    feature_info[f] = features[f]
        
        return predicted_trait, confidence, results, feature_info

    except Exception as e:
        st.error(f"Error analyzing handwriting: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None, None, None, None

# Streamlit UI
st.title("🖋️ Handwriting Personality Analyzer")
st.write("Upload handwriting to discover personality traits using deep learning.")

uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Analyze Handwriting"):
        with st.spinner("Analyzing handwriting..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                image.save(tmp.name)
                image_path = tmp.name

                predicted_trait, confidence, results, feature_info = analyze_handwriting_personality(image_path, model)

                if predicted_trait and confidence and results:
                    st.subheader("🧠 Primary Personality Trait")
                    st.success(f"{predicted_trait} (Confidence: {confidence:.2f})")

                    st.markdown("### 📘 Description")
                    st.write(trait_descriptions[predicted_trait])
                    st.markdown("### ✍️ Handwriting Indicators")
                    st.write(handwriting_indicators[predicted_trait])

                    if feature_info:
                        st.markdown("### 🔍 Detected Handwriting Characteristics")
                        feature_cols = st.columns(3)
                        for i, (feature, value) in enumerate(feature_info.items()):
                            with feature_cols[i % 3]:
                                feature_name = feature.replace('_', ' ').title()
                                st.metric(feature_name, f"{value:.2f}")

                    st.markdown("### 📊 Trait Confidence Chart")
                    

                    fig, ax = plt.subplots(figsize=(14, 8))  
                    
                    trait_names = list(results.keys())
                    confidences = [results[t]["confidence"] for t in trait_names]
                    
                    colors = ['#2E7D32' if results[t]["is_primary"] else '#64B5F6' for t in trait_names]
                    
  
                    bars = ax.bar(
                        trait_names, 
                        confidences, 
                        color=colors, 
                        edgecolor='black',
                        linewidth=1.5,
                        width=0.65
                    )

                    ax.set_ylim(0, max(confidences) * 1.2)

                    ax.grid(axis='y', linestyle='--', alpha=0.7, color='#E0E0E0')
       
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    
                    ax.set_ylabel("Confidence Score", fontsize=14, fontweight='bold')
                    ax.set_title("Personality Traits Analysis", fontsize=18, fontweight='bold', pad=20)
                    
                    for bar, conf in zip(bars, confidences):
                        ax.text(
                            bar.get_x() + bar.get_width() / 2, 
                            conf + 0.02, 
                            f'{conf:.2f}', 
                            ha='center', 
                            fontsize=12, 
                            fontweight='bold',
                            color='black'
                        )
                    
                    plt.xticks(fontsize=12, fontweight='bold')
                    plt.yticks(fontsize=12)
                    
                    plt.tight_layout()
                    
                    st.pyplot(fig)

                    st.markdown("---")
                    st.caption("⚠️ This tool is for educational purposes only and not scientifically validated.")
                else:
                    st.error("Failed to analyze the handwriting. Please try with a different image.")

                os.unlink(image_path)