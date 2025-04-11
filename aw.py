from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load GPT-Neo model and tokenizer
model_name = "EleutherAI/gpt-neo-1.3B"  # This is a 1.3 billion parameter model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Function to generate response
def bestie_reply(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=200,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(output[0], skip_special_tokens=True)

    # Keep only the part generated after user input
    return response[len(prompt):].strip()

# Start chat loop
print("Chat started with your bestie (type 'exit' to stop):")
conversation = ""

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Bestie: Bye bestie! Catch you later!")
        break

    # Add to conversation context
    conversation += f"You: {user_input}\nBestie: "
    reply = bestie_reply(conversation)
    print("Bestie:", reply)

    # Continue with context
    conversation += reply + "\n"