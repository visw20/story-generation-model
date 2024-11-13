# from transformers import T5ForConditionalGeneration, T5Tokenizer

# # Load the T5 model and tokenizer (using 't5-large' for better performance; you can also use 't5-small' or 't5-base')
# model = T5ForConditionalGeneration.from_pretrained("t5-large")
# tokenizer = T5Tokenizer.from_pretrained("t5-large")

# # Function to generate text
# def generate_text(prompt, max_length=100, num_beams=5):
#     # Prepend the task name (optional but helps guide T5)
#     input_text = "generate: " + prompt
    
#     # Tokenize the input text
#     inputs = tokenizer(input_text, return_tensors="pt")
    
#     # Generate the text with beam search for better quality
#     outputs = model.generate(
#         inputs.input_ids,
#         max_length=max_length,
#         num_beams=num_beams,
#         no_repeat_ngram_size=2,   # Avoid repeating phrases
#         early_stopping=True       # Stop early if all beams converge
#     )
    
#     # Decode the generated tokens to a string
#     generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return generated_text

# # Example 1: Story Writing
# story_prompt = "Write a story about a spaceship traveling to a new galaxy."
# print("Story Generation Output:\n", generate_text(story_prompt))

# # Example 2: Dialogue Generation
# dialogue_prompt = "Respond to: 'What are your favorite science fiction books?'"
# print("\nDialogue Generation Output:\n", generate_text(dialogue_prompt))

# # Example 3: Code Generation
# code_prompt = "Write a Python function to calculate the factorial of a number."
# print("\nCode Generation Output:\n", generate_text(code_prompt))





# from transformers import AutoModelForCausalLM, AutoTokenizer

# # Load the Qwen-2.5B Instruct model and tokenizer
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

# # Function to generate text with optimized parameters
# def generate_text(prompt, max_length=300, num_beams=5, temperature=0.7, top_k=50):
#     # Use a descriptive prompt that emphasizes storytelling
#     input_text = "Write an inspiring story about: " + prompt
    
#     # Tokenize the input text
#     inputs = tokenizer(input_text, return_tensors="pt")
    
#     # Generate text with settings optimized for storytelling
#     outputs = model.generate(
#         inputs.input_ids,
#         max_length=max_length,
#         num_beams=num_beams,
#         no_repeat_ngram_size=3,    # Reduce repetitive phrases
#         early_stopping=True,
#         temperature=temperature,   # Controls randomness for creativity
#         top_k=top_k                # Limits sampling to top-k options
#     )
    
#     # Decode and return the generated text
#     generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return generated_text

# # Example: Story Generation with a simplified prompt
# story_prompt = "Trump was won US president election"
# print("Story Generation Output:\n", generate_text(story_prompt))




from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the Qwen-2.5B Instruct model and tokenizer
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

# Function to generate text with optimized parameters
def generate_text(prompt, max_length=500, num_beams=5, temperature=0.8, top_k=50):
    # Enhance the prompt to guide the model in creating a complete story
    input_text = "Write a detailed and inspiring story about: " + prompt + " The journey of overcoming obstacles, challenges, and victory."
    
    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt")
    
    # Generate text with settings optimized for storytelling
    outputs = model.generate(
        inputs.input_ids,
        max_length=max_length,
        num_beams=num_beams,
        no_repeat_ngram_size=3,    # Reduce repetitive phrases
        early_stopping=True,
        temperature=temperature,   # Controls randomness for creativity
        top_k=top_k                # Limits sampling to top-k options
    )
    
    # Decode and return the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Example: Story Generation with an expanded prompt
story_prompt = "monkey win the banana game"
print("Story Generation Output:\n", generate_text(story_prompt))

