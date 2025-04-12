import streamlit as st
import os
import tempfile
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.models import load_model
import gdown
import numpy as np

MODEL_PATH = "handwriting_personality_model_final.h5"
GDRIVE_FILE_ID = "1-1u1rUYd8CC5HZrIu6Js8v2hhZvP_-H1"  

@st.cache_resource
def load_model_cached():
    if not os.path.exists(MODEL_PATH):
        gdown.download(f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}", MODEL_PATH, quiet=False)
    return load_model(MODEL_PATH)

model = load_model_cached()


trait_descriptions = {
    "Openness": "People high in Openness are curious, imaginative, and open to new experiences. They tend to enjoy art, new ideas, and adventure. High openness is often linked to unconventional thinking and creativity.",
    "Conscientiousness": "Highly conscientious individuals are organized, detail-oriented, and responsible. They tend to have strong self-discipline, act dutifully, and aim for achievement. Their handwriting often reflects structure and consistency.",
    "Extraversion": "Extraverts are outgoing, enthusiastic, and sociable. They seek stimulation in the company of others and may display energetic, talkative behavior. Their handwriting often appears large and expressive.",
    "Agreeableness": "Agreeable individuals are compassionate, cooperative, and trusting. They tend to avoid conflict and are often helpful and generous. Their handwriting is often smooth, rounded, and gentle in pressure.",
    "Neuroticism": "People with high neuroticism tend to experience emotional instability, anxiety, and mood swings. Their handwriting may show variability in slant, heavy pressure, or irregular patterns due to internal tension."
}

handwriting_indicators = {
    "Openness": "Wide spacing, artistic flourishes, inconsistent baselines, large loops in letters such as 'g' and 'y'.",
    "Conscientiousness": "Consistent slant, even spacing between letters and words, clear and legible characters, well-aligned text.",
    "Extraversion": "Large letter size, upward slant, wide loops in letters, fast writing speed, expressive strokes.",
    "Agreeableness": "Rounded letters, connected strokes, gentle pressure, slight right slant, overall harmony in script.",
    "Neuroticism": "Variable slant, sharp angles, heavy pressure, irregular spacing, tremors or uneven lines."
}

def analyze_handwriting_personality(image_path, model):
    try:
        # Preprocess image
        img = Image.open(image_path)
        img = img.resize((224, 224))  # Adjust size to match your model's input requirements
        img = img.convert('RGB')
        img_array = np.array(img)
        img_array = img_array / 255.0  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0)

        # Get model predictions
        predictions = model.predict(img_array, verbose=0)
        
        # Map predictions to traits
        trait_names = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
        results = {}
        
        for trait, pred in zip(trait_names, predictions[0]):
            results[trait] = {
                "confidence": float(pred),
                "is_primary": False
            }
        
        # Determine primary trait
        predicted_trait = max(results, key=lambda k: results[k]["confidence"])
        results[predicted_trait]["is_primary"] = True
        confidence = results[predicted_trait]["confidence"]
        
        return predicted_trait, confidence, results
        
    except Exception as e:
        st.error(f"Error analyzing handwriting: {str(e)}")
        return None, None, None

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

                predicted_trait, confidence, results = analyze_handwriting_personality(image_path, model)
                
                if predicted_trait and confidence and results:
                    st.subheader("🧠 Primary Personality Trait")
                    st.success(f"{predicted_trait} (Confidence: {confidence:.2f})")

                    st.markdown("### 📘 Description")
                    st.write(trait_descriptions[predicted_trait])
                    st.markdown("### ✍️ Handwriting Indicators")
                    st.write(handwriting_indicators[predicted_trait])

                    # Bar chart visualization
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
                else:
                    st.error("Failed to analyze handwriting. Please try another image.")
                
                os.unlink(image_path)