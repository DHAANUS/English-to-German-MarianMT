# English → German Machine Translation (Eng2Ger)

This project demonstrates fine-tuning of **MarianMT (Helsinki-NLP/opus-mt-en-de)** for **English → German translation** using Hugging Face Transformers.

---

## Project Overview
- **Model**: MarianMT (pretrained seq2seq model for machine translation).  
- **Dataset**: Custom parallel corpus of English–German sentence pairs (TSV file).  
- **Task**: Translate English text into German.  
- **Framework**: Hugging Face Transformers + PyTorch + W&B logging.  

---

## Key Features
- Text preprocessing (lowercasing, punctuation cleaning).  
- Custom dataset conversion to Hugging Face `Dataset`.  
- Fine-tuning pretrained MarianMT with `Trainer`.  
- W&B integration for logging experiments.  
- Inference script to translate arbitrary text.  

---

## Training Workflow
1. Load dataset from TSV file.  
2. Clean and tokenize sentences.  
3. Train with Hugging Face `Trainer`.  
4. Evaluate on held-out validation split.  
5. Save fine-tuned model + tokenizer for inference.  

Example training setup:
```python
from transformers import AutoModelForSeq2SeqLM, Trainer, TrainingArguments

model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-de")

training_args = TrainingArguments(
    output_dir='./results',
    learning_rate=5e-5,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=True,
    evaluation_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_test_split["train"],
    eval_dataset=train_test_split["test"],
)

trainer.train()
