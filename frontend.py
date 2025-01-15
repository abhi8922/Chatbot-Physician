import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# Load the trained model and tokenizer
model_path = "./fine_tuned_biobert"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForQuestionAnswering.from_pretrained(model_path)

# Define a function for prediction
def predict_answer(question, context):
    inputs = tokenizer(question, context, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    start_index = torch.argmax(outputs.start_logits)
    end_index = torch.argmax(outputs.end_logits)

    answer_tokens = inputs.input_ids[0][start_index:end_index + 1]
    return tokenizer.decode(answer_tokens)

# Streamlit UI
st.title("Medical Question Answering System")
st.write("Ask a question based on the given context.")

# User inputs for question and context
question = st.text_input("Enter your question:")
context = st.text_area("Enter the context:")

# Display the answer when user clicks the button
if st.button("Get Answer"):
    if question and context:
        answer = predict_answer(question, context)
        st.write(f"**Answer:** {answer}")
    else:
        st.write("Please provide both a question and context.")
