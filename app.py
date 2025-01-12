# Importing the requirements
import warnings
warnings.filterwarnings("ignore")

import gradio as gr
from src.paligemma.response import caption_image


# Image and input parameters
image = gr.Image(type="pil", label="Image")
max_new_tokens = gr.Slider(
    minimum=20,
    maximum=260,
    value=120,
    step=20,
    label="Max Tokens",
    info="Use larger values for detailed captions",
)
language = gr.Dropdown(
    choices=["English", "Spanish", "French"],
    value="English",
    label="Language",
    info="Select the caption language",
)

# Output for the interface
answer = gr.Textbox(label="Generated Caption", show_label=True, show_copy_button=True)

# Examples for the interface
examples = [
    ["images/cat.jpg", 140, "Spanish"],
    ["images/dog.jpg", 260, "English"],
    ["images/bird.jpg", 120, "French"],
]

# Title, description, and article for the interface
title = "PaliGemma Image Captioning"
description = "Gradio Demo for the PaliGemma 2 Vision Language Understanding and Generation model. This model generates natural language captions based on uploaded images. To use it, upload your image, select the desired parameters (or stick with the default settings), and click 'Submit.' You can also choose one of the examples to load a predefined image. For more information, please refer to the links below."
article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2412.03555' target='_blank'>PaliGemma 2: A Family of Versatile VLMs for Transfer</a> | <a href='https://huggingface.co/google/paligemma2-3b-ft-docci-448' target='_blank'>Model Page</a></p>"


# Launch the interface
interface = gr.Interface(
    fn=caption_image,
    inputs=[image, max_new_tokens, language],
    outputs=answer,
    examples=examples,
    cache_examples=True,
    cache_mode="lazy",
    title=title,
    description=description,
    article=article,
    theme="Monochrome",
    flagging_mode="never",
)
interface.launch(debug=False)
