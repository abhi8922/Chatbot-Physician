from transformers import GPT2LMHeadModel,GPT2Tokenizer

model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer= GPT2Tokenizer.from_pretrained(model_name)

input_text = "Once upon a time"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

output = model.generate(input_ids,max_length=50, num_return_sequences=1)

generated_text = tokenizer.decode(output[0],skip_special_tokens= True)
print(generated_text)
