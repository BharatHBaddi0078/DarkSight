import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
import torch
import ssl

# Option 1: Bypass SSL verification (temporary fix, not recommended for production)
ssl._create_default_https_context = ssl._create_unverified_context

# Option 2: Set Hugging Face offline mode if needed (not preferred unless manual downloads are in place)
# os.environ['TRANSFORMERS_OFFLINE'] = '1'

# If you downloaded the tokenizer manually, provide the local path here:
# tokenizer = BertTokenizer.from_pretrained('/path/to/local/bert-base-uncased')

# Otherwise, load tokenizer from Hugging Face (works if SSL bypass is successful)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load your dataset
data = pd.read_csv('data.csv')
texts = data['text'].tolist()
labels = data['label'].tolist()

# Split the dataset into training and testing sets
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Tokenize the data
def tokenize_function(texts):
    return tokenizer(texts, padding="max_length", truncation=True, max_length=512)

train_encodings = tokenizer(train_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
test_encodings = tokenizer(test_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")

# Create a custom Dataset class for PyTorch
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Create the training and testing datasets
train_dataset = TextDataset(train_encodings, train_labels)
test_dataset = TextDataset(test_encodings, test_labels)

# Load pre-trained BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # number of training epochs
    per_device_train_batch_size=8,   # batch size for training
    per_device_eval_batch_size=8,    # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,                # log every 10 steps
    evaluation_strategy="epoch"      # evaluation strategy to run at the end of each epoch
)

# Define a data collator
data_collator = DataCollatorWithPadding(tokenizer)

# Initialize the Trainer
trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments
    train_dataset=train_dataset,         # training dataset
    eval_dataset=test_dataset,           # evaluation dataset
    tokenizer=tokenizer,                 # tokenizer used for processing the dataset
    data_collator=data_collator          # data collator to handle padding
)

# Train the model
trainer.train()

# Evaluate the model on the test set
predictions = trainer.predict(test_dataset)
preds = predictions.predictions.argmax(-1)

# Print the evaluation metrics
print("Accuracy:", accuracy_score(test_labels, preds))
print("Classification Report:\n", classification_report(test_labels, preds))

# Save the fine-tuned model and tokenizer
model.save_pretrained('./fine_tuned_model')
tokenizer.save_pretrained('./fine_tuned_model')
