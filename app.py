# Importing the requirements
import warnings
warnings.filterwarnings("ignore")

import gradio as gr
from src.app.response import caption_image


# Image and input parameters
image = gr.Image(type="pil", label="Image")
max_new_tokens = gr.Slider(
    minimum=20,
    maximum=160,
    value=80,
    step=10,
    label="Max Tokens",
    info="Use larger values for detailed captions",
)
sampling = gr.Checkbox(value=False, label="Sampling")

# Output for the interface
answer = gr.Textbox(label="Generated Caption", show_label=True, show_copy_button=True)

# Examples for the interface
examples = [
    ["images/cat.jpg", 100, False],
    ["images/dog.jpg", 80, True],
    ["images/bird.jpg", 160, False],
]

# Title, description, and article for the interface
title = "PaliGemma 2 Image Captioning"
description = "Gradio Demo for the PaliGemma 2 Vision Language Understanding and Generation model. This model generates natural language captions based on uploaded images. To use it, upload your image, select the desired parameters (or stick with the default settings), and click 'Submit.' You can also choose one of the examples to load a predefined image. For more information, please refer to the links below."
article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2412.03555' target='_blank'>Model Paper</a> | <a href='https://huggingface.co/google/paligemma2-3b-ft-docci-448' target='_blank'>Model Page</a></p>"


# Launch the interface
interface = gr.Interface(
    fn=caption_image,
    inputs=[image, max_new_tokens, sampling],
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
