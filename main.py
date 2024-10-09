from fastapi import FastAPI
from pydantic import BaseModel
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from langdetect import detect, DetectorFactory

# Set seed for language detection stability
DetectorFactory.seed = 0

# Load the model and tokenizer
model_name = "facebook/m2m100_418M"
tokenizer = M2M100Tokenizer.from_pretrained(model_name)
model = M2M100ForConditionalGeneration.from_pretrained(model_name)

# Initialize FastAPI app
app = FastAPI()

# Set target language
target_lang = "en"  # English

# Define the request body using Pydantic
class TranslationRequest(BaseModel):
    text: str

# Define the welcome endpoint (root route)
@app.get("/")
async def welcome():
    return {"message": "Welcome to the Translation API! Use the /translate endpoint to translate text to English."}

# Define the translation endpoint
@app.post("/translate")
async def translate_text(request: TranslationRequest):
    input_text = request.text

    # Detect the language of the input text
    detected_lang = detect(input_text)

    # Check if the detected language is already English
    if detected_lang == target_lang:
        return {"detected_language": detected_lang, "translated_text": input_text}

    # Tokenize the input and generate translation
    tokenizer.src_lang = detected_lang
    encoded_input = tokenizer(input_text, return_tensors="pt")
    generated_tokens = model.generate(**encoded_input, forced_bos_token_id=tokenizer.get_lang_id(target_lang))

    # Decode the translated text
    translated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

    # Return the detected language and the translated text
    return {"detected_language": detected_lang, "translated_text": translated_text}


# Run the FastAPI app with Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


# 日本の食べ物は美味しいです。
# Je suis étudiant
# السلام عليكم