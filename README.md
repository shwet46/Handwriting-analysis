# 🖋️ Handwriting Personality Analyzer

A deep learning-powered web application that analyzes handwriting samples to predict personality traits based on the Big Five (OCEAN) model.

## 🎯 Overview

This application uses machine learning to analyze handwriting characteristics and predict personality traits. It processes handwriting samples through a trained neural network to estimate the writer's personality dimensions based on the Five Factor Model (OCEAN).

## 📊 Dataset

This project uses the [Handwriting Recognition Dataset](https://www.kaggle.com/landlord/handwriting-recognition) from Kaggle. The dataset contains:
- Handwriting samples
- Associated personality trait scores
- Various writing styles and characteristics

## 🎮 Features

- **Image Upload**: Support for JPG, JPEG, and PNG formats.
- **Feature Analysis**: 
  - Size and Scale
  - Slant Analysis
  - Pressure Detection
  - Spacing Measurement
  - Line Consistency
  - Pressure Variability
- **Personality Predictions**:
  - Openness
  - Conscientiousness
  - Extraversion
  - Agreeableness
  - Neuroticism
- **Visualization**: Interactive charts and metrics.


## 🤖 Model Architecture

- Input: Grayscale images (400x150 pixels)
- Feature Extraction: Convolutional layers
- Classification: Dense layers
- Output: 5 personality trait scores

## 📊 Analysis Pipeline

1. Image Preprocessing
   - Grayscale conversion
   - Resize to 400x150
   - Normalization
2. Feature Extraction
   - Size analysis
   - Slant detection
   - Pressure analysis
3. Personality Prediction
   - Trait confidence scoring
   - Feature-based adjustments
   - Final personality profile

## ⚠️ Disclaimer

This tool is for educational and entertainment purposes only. The predictions should not be used for clinical assessment or decision-making.

---
