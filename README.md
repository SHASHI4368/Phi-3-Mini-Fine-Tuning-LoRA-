# Phi-3 Mini Fine-Tuning (LoRA) 🚀  

This repository contains code and experiments for fine-tuning the **[microsoft/Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)** model using **Parameter-Efficient Fine-Tuning (PEFT)** with **LoRA**.  
The project is designed for **instruction tuning** and generating **product advertisements** with natural language generation.  

---

## 📌 Features
- ✅ Fine-tunes `Phi-3-mini` with **LoRA adapters**  
- ✅ Uses **Hugging Face Transformers** + **PEFT** for training  
- ✅ Dataset handling via **🤗 Datasets**  
- ✅ Experiment tracking with **MLflow**  
- ✅ Model export & inference pipeline with Hugging Face `pipeline`  
- ✅ API integration with **FastAPI**  

---

## 📂 Project Structure
```
├── phi3_mini_4_instruct_model_fine_tune.ipynb   # Main training notebook
├── requirements.txt                             # Dependencies
├── README.md                                    # Project documentation
└── /models                                      # Fine-tuned models & LoRA adapters
```

---

## ⚙️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
```

---

## 📊 Training

Open the notebook:

```bash
jupyter notebook phi3_mini_4_instruct_model_fine_tune.ipynb
```

Key steps inside:
1. Load tokenizer & base model (`microsoft/Phi-3-mini-4k-instruct`)
2. Prepare dataset (JSON/CSV → 🤗 `Dataset`)
3. Apply **LoRA** via PEFT
4. Train with Hugging Face `Trainer`
5. Log metrics with **MLflow**
6. Save model & adapter  

---

## 🚀 Inference

After training, you can load the fine-tuned model for text generation:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel

base_model = "microsoft/Phi-3-mini-4k-instruct"
lora_model = "./models/phi3-product-ads-lora"

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base_model)
model = PeftModel.from_pretrained(model, lora_model)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

print(generator("Write an engaging ad for a smartwatch", max_new_tokens=100))
```

---

## 🌐 API Deployment (FastAPI)

You can serve the model with a **REST API**:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Example `app.py` snippet:

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class ProductRequest(BaseModel):
    product_name: str
    product_description: str

@app.post("/generate-ad")
def generate_ad(request: ProductRequest):
    # Call your pipeline here
    return {"ad_text": f"Generated ad for {request.product_name}"}
```

---

## 📈 Experiment Tracking

Training runs are tracked with **MLflow**:

```bash
mlflow ui
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) to view metrics, losses, and artifacts.

---

## 🛠 Requirements

- Python 3.10+
- PyTorch
- Hugging Face Transformers
- Datasets
- PEFT
- Accelerate
- MLflow
- FastAPI (for API deployment)

Install everything via:

```bash
pip install -r requirements.txt
```

---

## 📜 License

This project is licensed under the **MIT License**.  

---

## 🙌 Acknowledgements
- [Microsoft Phi-3](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)  
- [Hugging Face Transformers](https://github.com/huggingface/transformers)  
- [PEFT (LoRA)](https://github.com/huggingface/peft)  
- [MLflow](https://mlflow.org/)  
