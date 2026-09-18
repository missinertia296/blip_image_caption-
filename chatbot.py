from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
model_name = "facebook/blenderbot-400M-distill"

# Load model (download on first run and reference local installation for subsequent runs)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
conversation_history = []
print("Chatbot ready! (type 'exit' to quit)\n")
history_string = "\n".join(conversation_history)
input_text = input("> ")
prompt = history_string + f"\nUser: {input_text}\nBot:"

inputs = tokenizer(
    prompt,
    return_tensors="pt",
    truncation=True,
    max_length=512
)
outputs = model.generate(
    **inputs,
    max_new_tokens=60,
    no_repeat_ngram_size=3,
    repetition_penalty=1.3,
    do_sample=True,
    temperature=0.6,
    top_p=0.85
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
print(response)




