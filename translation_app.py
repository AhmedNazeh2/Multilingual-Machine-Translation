import streamlit as st
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer # type: ignore
from langdetect import detect, DetectorFactory  # type: ignore

# Set seed for language detection stability
DetectorFactory.seed = 0

# Load the model and tokenizer
model_name = "facebook/m2m100_418M"
tokenizer = M2M100Tokenizer.from_pretrained(model_name)
model = M2M100ForConditionalGeneration.from_pretrained(model_name)

# Display a welcome message
st.success("Welcome to the Google Machine Translation")

# Streamlit app
st.title("Multilingual Text Translation")

st.write("This app detects the language of the input text and translates it to English.")

# Input text
input_text = st.text_area("Please enter text for translation:")

# Add a button for translation
if st.button("Translate"):  # Translation happens when the button is clicked
    if input_text:
        # Detect the language of the input text
        detected_lang = detect(input_text)
        st.write(f"Detected language: {detected_lang}")

        # Set the target language
        target_lang = "en"  # English

        # Check if the detected language is not English
        if detected_lang == target_lang:
            st.write("The text is already in English.")
            translated_text = input_text  # No translation needed
        else:
            # Tokenize the input and generate translation
            tokenizer.src_lang = detected_lang  # Set the detected source language
            encoded_input = tokenizer(input_text, return_tensors="pt")
            generated_tokens = model.generate(**encoded_input, forced_bos_token_id=tokenizer.get_lang_id(target_lang))

            # Decode the translated text
            translated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

        # Display the translated text
        st.write("Translated text:", translated_text)
    else:
        st.warning("Please enter some text to translate.")