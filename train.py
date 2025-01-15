# Required libraries
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
import torch

# Load the data
df = pd.read_csv("med.csv")

# Step 1: Prepare dataset - add context and answer_start columns
def prepare_dataset(df):
    # Ensure 'answer' and 'context' are strings and handle NaN values
    df['context'] = df['focus_area'].fillna("").astype(str)
    df['answer'] = df['answer'].fillna("").astype(str)
    
    # Calculate 'answer_start' by finding the position of 'answer' within 'context'
    df['answer_start'] = df.apply(
        lambda row: row['context'].find(row['answer']) if row['answer'] in row['context'] else -1, axis=1
    )
    
    # Filter out rows where answer_start couldn't be determined
    if df['answer_start'].eq(-1).any():
        print("Some answers were not found in the context and will be removed.")
    df = df[df['answer_start'] != -1].reset_index(drop=True)
    return df

# Prepare the DataFrame
df = prepare_dataset(df)

# Ensure columns are as expected
required_columns = {'question', 'context', 'answer', 'answer_start'}
if not required_columns.issubset(df.columns):
    raise ValueError(f"Dataset is missing required columns. Expected columns: {required_columns}")

# Step 2: Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# Load BioBERT model and tokenizer
model_name = "dmis-lab/biobert-v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Step 3: Define preprocessing function for tokenization and answer position
def preprocess_function(examples):
    inputs = tokenizer(
        examples["question"],
        examples["context"],
        max_length=384,
        truncation="only_second",
        padding="max_length",
        return_offsets_mapping=True
    )
    
    start_positions = []
    end_positions = []

    for i, offset_mapping in enumerate(inputs["offset_mapping"]):
        start_char = examples["answer_start"][i]
        end_char = start_char + len(examples["answer"][i])

        # Initialize token positions
        start_token, end_token = 0, 0
        for idx, (start, end) in enumerate(offset_mapping):
            if start <= start_char < end:
                start_token = idx
            if start < end_char <= end:
                end_token = idx
                break

        start_positions.append(start_token)
        end_positions.append(end_token)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    inputs.pop("offset_mapping")  # Remove offset mapping after processing
    
    return inputs

# Apply preprocessing
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Step 4: Set up training arguments
training_args = TrainingArguments(
    output_dir="./fine_tuned_biobert",
    eval_strategy="no",  # Disable evaluation since there's no eval dataset provided
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
)

# Step 5: Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Train the model
trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained("./fine_tuned_biobert")
tokenizer.save_pretrained("./fine_tuned_biobert")

print("Fine-tuning complete. Model saved to ./fine_tuned_biobert")
