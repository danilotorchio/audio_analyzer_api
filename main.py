import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

model = tf.keras.models.load_model("lia_model.h5")
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

def prepare_data(input_text, tokenizer):
    token = tokenizer.encode_plus(
        input_text,
        max_length=256,
        truncation=True,
        padding='max_length',
        add_special_tokens=True,
        return_tensors='tf'
    )
    return {
        'input_ids': tf.cast(token.input_ids, tf.float64),
        'attention_mask': tf.cast(token.attention_mask, tf.float64)
    }

def predict(text: str):
    pred = model.predict(prepare_data(text, tokenizer))
    out = np.argmax(pred)

    match out:
        case 0:
            return "Negativo"
        case 1:
            return "Positivo"
        case 2:
            return "Neutro"
        case _:
            return "Error"


class DataBody(BaseModel):
    text: str

@app.post("/")
async def analyze(data: DataBody):
    sentiment = predict(data.text)
    return {"message": data.text, "sentiment": sentiment}