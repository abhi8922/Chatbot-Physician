# Import necessary libraries
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score

# Step 1: Prepare training dataset without `context`
def prepare_dataset(df):
    df['answer'] = df['answer'].fillna("").astype(str)
    df['combined_answer'] = "Answer: " + df['answer']
    df['answer_start'] = 0  # Default since context is not present
    return df[['question', 'combined_answer', 'answer_start']]

# Load the training data
df = pd.read_csv("med.csv")
print(df.columns)  # Check if 'context' is missing
df = prepare_dataset(df)

# Prepare dataset for training
dataset = Dataset.from_pandas(df.rename(columns={"combined_answer": "answer"}))

# Step 2: Load BioBERT model and tokenizer
model_name = "dmis-lab/biobert-v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Step 3: Preprocessing function for training
def preprocess_function(examples):
    inputs = tokenizer(
        examples["question"],
        examples["answer"],
        max_length=384,
        truncation="only_second",
        padding="max_length",
        return_offsets_mapping=True
    )
    inputs["start_positions"] = [0] * len(examples["answer"])  # No positional mapping needed
    inputs["end_positions"] = [len(inputs.input_ids[i]) - 1 for i in range(len(inputs.input_ids))]
    inputs.pop("offset_mapping")
    return inputs

# Apply preprocessing for training
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Step 4: Set up training arguments
training_args = TrainingArguments(
    output_dir="./fine_tuned_biobert",
    eval_strategy="no",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
)

# Step 5: Initialize Trainer and train the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained("./fine_tuned_biobert")
tokenizer.save_pretrained("./fine_tuned_biobert")
print("Fine-tuning complete. Model saved to ./fine_tuned_biobert")

# Step 6: Automated testing
# Load the fine-tuned model and tokenizer
model = AutoModelForQuestionAnswering.from_pretrained("./fine_tuned_biobert")
tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_biobert")

# Define a predefined test set
test_data = {
    "question": ["What is the treatment for diabetes?", "What is the common cause of fever?"],
    "answer": ["medication, lifestyle changes, and insulin injections", "infection, often caused by viruses or bacteria"]
}

# Load test data into DataFrame
test_df = pd.DataFrame(test_data)
test_df["combined_answer"] = "Answer: " + test_df["answer"]

# Prediction function for testing
def predict_answer(question, combined_answer):
    inputs = tokenizer(question, combined_answer, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    start_index = torch.argmax(outputs.start_logits)
    end_index = torch.argmax(outputs.end_logits)

    answer_tokens = inputs.input_ids[0][start_index:end_index + 1]
    return tokenizer.decode(answer_tokens)

# Evaluate on predefined test set
predicted_answers = []

for i in range(len(test_df)):
    question = test_df.iloc[i]["question"]
    combined_answer = test_df.iloc[i]["combined_answer"]
    predicted_answer = predict_answer(question, combined_answer)
    predicted_answers.append(predicted_answer)

# Calculate accuracy and F1 score
true_answers = test_df["combined_answer"].tolist()
accuracy = accuracy_score(true_answers, predicted_answers)
f1 = f1_score(true_answers, predicted_answers, average="weighted")

# Print results
for i in range(len(test_df)):
    print(f"Question: {test_df.iloc[i]['question']}")
    print(f"Predicted Answer with Context:\n{predicted_answers[i]}")
    print(f"True Answer with Context:\n{true_answers[i]}")
    print("-" * 40)

print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1}")
