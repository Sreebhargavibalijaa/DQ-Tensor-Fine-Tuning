# code will be added soon
# DQ-Tensor-Fine-Tuning
DQ Tensor Fine-Tuning is a novel approach to efficiently fine-tune transformer models by integrating dynamically quantized tensors into the classification layer. This technique enhances training efficiency and model adaptability for various NLP tasks.

1) Uses dynamically quantized tensors for efficient parameter updates

2) Supports BERT-based transformer models

3) Designed for classification tasks with configurable label outputs


Usage
from dq_tensor_finetuning import DQTensorFineTuning
from transformers import AutoTokenizer
import torch

model_name = "bert-base-uncased"
num_labels = 2
model = DQTensorFineTuning(model_name, num_labels)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Example input
tokens = tokenizer("Example input text", return_tensors="pt")
input_ids = tokens["input_ids"]
attention_mask = tokens["attention_mask"]

# Forward pass
output = model(input_ids, attention_mask)
print(output)
