from flask import Flask, request, jsonify, send_from_directory
import torch
import torch.nn.functional as F
from transformers import BertTokenizer

app = Flask(__name__)

model = torch.load("sentiment_analysis_model.pt", map_location=torch.device('cpu'))
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

@app.route("/")
def home():
    return send_from_directory(directory=".", filename="home.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form["text"]
    encoded_text = tokenizer.encode_plus(
        text,
        max_length=128,
        add_special_tokens=True,
        pad_to_max_length=True,
        return_attention_mask=True
    )
    input_ids = torch.tensor([encoded_text["input_ids"]])
    attention_mask = torch.tensor([encoded_text["attention_mask"]])
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        logits = outputs.logits
        prediction = F.softmax(logits, dim=1).tolist()[0]
        sentiment = "positive" if prediction[1] > prediction[0] else "negative"
    return f"<h1>Prediction: {sentiment}</h1>"

if __name__ == "__main__":
    app.run(debug=True)
