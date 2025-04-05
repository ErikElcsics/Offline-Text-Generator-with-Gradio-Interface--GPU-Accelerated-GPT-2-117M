# 🚀 Offline Text Generator with Gradio Interface - GPU-Accelerated GPT-2 (117M)

This repository contains a simple app that uses **GPT-2** for text generation. It leverages **Gradio** for the user interface and optimizes inference with **GPU acceleration**. The app can generate creative and contextually relevant text based on the provided prompt. The model runs offline, making it highly efficient for local use.

## 🌟 Features

- **GPU-Accelerated Inference**: The app utilizes your system's GPU (if available) for faster and more efficient text generation.
- **Gradio Interface**: A simple and interactive user interface built with Gradio for seamless text input and output.
- **Offline Operation**: The model runs offline, providing privacy and fast response times for text generation.
- **Text Generation**: Generate text responses based on any user-provided prompt.
- **Customizable Generation Parameters**: Tweak the text generation process with options like temperature and top-k sampling for varied responses.

## 🧠 Model Information

This app uses **GPT-2**, a powerful language model developed by OpenAI. Specifically, it uses the **117M parameter** version of GPT-2, which strikes a balance between computational efficiency and performance. The model is pre-trained on large-scale text data and can generate human-like text based on the input prompt.

## 📝 How the Code Works - Summary

1. **Model Loading**: The GPT-2 model and tokenizer are loaded into memory using the `transformers` library.
2. **GPU Optimization**: If a GPU is available, the model is moved to the GPU to speed up inference.
3. **Text Generation**: The model generates text based on a user-provided prompt, which is then returned through the Gradio interface.
4. **Gradio Interface**: The app uses Gradio to create an interactive UI where users can input prompts and receive generated text.


## 📦 Installation Instructions

To run this app locally, follow these installation steps:

1. Clone the repository:
   
   git clone https://github.com/ErikElcsics/Offline-Text-Generator-with-Gradio-Interface--GPU-Accelerated-GPT-2-117M.git
   cd Offline-Text-Generator
   

2. Install the necessary Python packages:
   
   - To install the necessary dependencies, run the following pip commands:

	- pip install gradio requests transformers torch

   - For GPU-enabled versions of PyTorch, run (change to match your GPU):

	- Example - pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

   
## 🔧 Setup Instructions

3. Run the app:
   
   python OfflineTextGeneratorGradio.py

Screen will show below and give you the url to launch your web app:
🚀 Using device: CUDA
🧠 Loading model...
* Running on local URL:  http://127.0.0.1:7860

Example:

![image](https://github.com/user-attachments/assets/85e1b7a3-a5dc-4166-897c-03348e8d5bc2)   

This will start the Gradio interface and allow you to interact with the text generation model.

## 🎮 How to Use the App

1. **Start the App**: After following the installation instructions, run the app by executing:
   
   python OfflineTextGeneratorGradio.py
   
2. **Interact with the Model**: 
   - Open the app in your browser, where you'll be greeted with an interface that allows you to input a text prompt.
   - Enter your prompt in the **Textbox** input field and click **Submit**.
   
3. **Get the Generated Text**: The model will generate a response based on your input, which will appear in the output field below.

## 🧰 Libraries Used

- [`gradio`](https://gradio.app/): A simple interface library to create interactive UIs.
- [`transformers`](https://huggingface.co/transformers/): A library for working with pre-trained models like GPT-2.
- [`torch`](https://pytorch.org/): Framework used for deep learning tasks and GPU acceleration.
- [`requests`](https://requests.readthedocs.io/): Python HTTP library (although not specifically used here, it’s included as part of the requirements).

## 💡 App Summary

### From a User Perspective
- **Text Prompt**: The user inputs a text prompt in the provided textbox.
- **Text Generation**: Upon submitting the prompt, the app generates a response based on the GPT-2 model.
- **Output**: The generated text is displayed in real-time as the app uses a Gradio interface.

### From a Technical Perspective
- **Model Loading**: The GPT-2 model is loaded from the Hugging Face `transformers` library.
- **GPU Utilization**: The app detects the availability of a GPU and moves the model to the GPU for faster text generation.
- **Text Generation**: The text is generated by encoding the user’s prompt, passing it through the GPT-2 model, and decoding the result.
- **Gradio UI**: The Gradio library is used to create an easy-to-use interface for users to interact with the model.

## ❤️ Credits

- Based on OpenAI’s GPT-2 model
- Interface built with [Gradio](https://gradio.app/)

![image](https://github.com/user-attachments/assets/1ea10797-4e32-4714-ab2d-4a22ed3205bb)

## Additional Information Gradio

Gradio Out of Box Settings

![image](https://github.com/user-attachments/assets/15d35b8e-f480-46f8-b62f-d5585f057040)

![image](https://github.com/user-attachments/assets/cb9b6669-f693-45ca-8cde-627d611dddfe)

## Information on Gradio

Gradio provides a simple and intuitive way to create interactive UIs for machine learning models. Out-of-the-box, Gradio offers the following default settings:

- Auto-layout: Gradio automatically arranges inputs and outputs in a clean, user-friendly layout without needing extra configuration.
- Support for Multiple Input Types: It includes built-in support for different input types, such as text, images, audio, and video, without needing additional setup.
- Output Options: Supports various output types, such as text, images, audio, and video, for displaying model results.
- Minimal Setup: Users can launch interfaces with just a function, specifying inputs and outputs in a simple manner.
- Sharing: Gradio allows for easy sharing of interfaces by providing a shareable link, making it accessible to others without hosting.
- Mobile Responsiveness: Interfaces are automatically optimized for mobile and desktop use.
- Pre-configured Interface: The interface is designed to work with a variety of machine learning frameworks (like TensorFlow, PyTorch, Hugging Face) out-of-the-box.
These settings make Gradio an efficient tool for quickly building interactive demos for machine learning models.

