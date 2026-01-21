import streamlit as st
from classifier import TextPredictor, extract_ordinal, extract_payment

# Page configuration
st.set_page_config(page_title="UltraLite Classifier Test", page_icon="🔍")

st.title("Classifier Model Tester")
st.markdown("Enter a message below to see how the model classifies it and extracts data.")

# Initialize model (cached so it only loads once)
@st.cache_resource
def load_model():
    try:
        return TextPredictor()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

predictor = load_model()

# User Input
user_input = st.text_area("Input Text:", placeholder="e.g., I want to pay with credit card or select the second option")

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter some text first.")
    elif predictor:
        with st.spinner("Classifying..."):
            # 1. Get Prediction
            label, confidence = predictor.predict(user_input)
            
            # 2. Extract specific values based on the logic in your classifier.py
            extracted_value = user_input
            if label == "Select_Option":
                extracted_value = extract_ordinal(user_input)
            elif label == "Make_Payment":
                extracted_value = extract_payment(user_input)

            # 3. Display Results
            st.divider()
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Predicted Class", label)
                st.metric("Confidence", f"{confidence:.2%}")
            
            with col2:
                st.subheader("Extracted Value")
                st.info(f"**{extracted_value}**")
                
            # Raw JSON view for debugging
            with st.expander("View Raw JSON"):
                st.json({
                    "input": user_input,
                    "class": label,
                    "confidence": float(confidence),
                    "extracted": extracted_value
                })