# Import required libraries
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import DataLoader, TensorDataset

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Define training data (text samples and their intent labels)
dataset = [
    ("hello", "greeting"),
    ("how are you", "greeting"),
    ("what is your name", "greeting"),
    ("who are you", "bot identity"),
    ("what can you do", "bot identity"),
    ("bye", "farewell"),
    ("tell me a joke", "joke"),
    ("what is the weather today", "weather"),
    ("What is artificial intelligence?", "general_info"),
    ("Who is Albert Einstein?", "general_info"),
    ("What is the capital of France?", "general_info"),
]

# Separate input text and labels
texts, labels = zip(*dataset)

# Convert label names to unique numerical values
label_map = {label: i for i, label in enumerate(set(labels))}
label_ids = [label_map[label] for label in labels]


# Tokenize input texts and convert them into numerical tensors
tokenized_input = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# Convert label IDs into tensors
label_ids = torch.tensor(label_ids)

# Prepare the dataset and create a DataLoader for training
dataset = TensorDataset(tokenized_input["input_ids"], tokenized_input["attention_mask"], label_ids)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Load a pre-trained BERT model and configure it for classification
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(label_map))
model.train()  # Set the model to training mode

# Define the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Train the model for multiple epochs
epochs = 5
for epoch in range(epochs):
    total_loss = 0
    for batch in dataloader:
        input_ids, attention_mask, labels = batch

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss / len(dataloader):.4f}")

# Save the trained model for later use
model.save_pretrained("bert_intent_model")
tokenizer.save_pretrained("bert_intent_model")

print("Training complete! Model saved in 'bert_intent_model/'")
