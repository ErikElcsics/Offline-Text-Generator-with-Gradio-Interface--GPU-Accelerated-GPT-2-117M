import os
import gradio as gr
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# ---- Settings ----
MODEL_NAME = "gpt2"  # 117M parameter model
MODEL_PATH = os.path.join("models", MODEL_NAME)

# ---- Check for GPU ----
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {device.upper()}")

# ---- Load Model ----
print("🧠 Loading model...")
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(device)  # Move model to GPU

# Save model locally if not exists
if not os.path.exists(MODEL_PATH):
    print("💾 Saving model locally...")
    model.save_pretrained(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH)


# ---- Optimized GPU Inference ----
def generate_text(prompt):
    # Encode and move to GPU
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Generate with GPU acceleration
    outputs = model.generate(
        inputs,
        max_length=200,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7,
        top_k=50
    )

    # Decode and return
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# ---- Gradio UI ----
iface = gr.Interface(
    fn=generate_text,
    inputs=gr.Textbox(label="Prompt", lines=4, placeholder="Ask GPT-2 anything..."),
    outputs=gr.Textbox(label="Answer - Generated Text"),
    title="🚀 Offline Text Generator with Gradio Interface - GPU-Accelerated GPT-2 (117M)",
    description=f"Running on {device.upper()} | Generates text with GPT-2 117M parameter model"
)

iface.launch()