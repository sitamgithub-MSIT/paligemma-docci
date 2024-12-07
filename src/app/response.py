# Necessary imports
import sys
import PIL.Image
import torch
import gradio as gr
import spaces

# Local imports
from src.config import device, model_name
from src.app.model import load_model_and_processor
from src.logger import logging
from src.exception import CustomExceptionHandling


# Model and processor
model, processor = load_model_and_processor(model_name, device)


@spaces.GPU
def caption_image(image: PIL.Image.Image, max_new_tokens: int, sampling: bool) -> str:
    """
    Generates a caption based on the given image using the model.

    Args:
        - image (PIL.Image.Image): The input image to be processed.
        - max_new_tokens (int): The maximum number of new tokens to generate.
        - sampling (bool): Whether to use sampling or not.

    Returns:
        str: The generated caption text.
    """
    try:
        # Check if image is None
        if not image:
            gr.Warning("Please provide an image.")

        # Prepare the inputs
        prompt = "caption en"
        model_inputs = (
            processor(text=prompt, images=image, return_tensors="pt")
            .to(torch.bfloat16)
            .to(device)
        )
        input_len = model_inputs["input_ids"].shape[-1]

        # Generate the response
        with torch.inference_mode():
            generation = model.generate(
                **model_inputs, max_new_tokens=max_new_tokens, do_sample=sampling
            )
            generation = generation[0][input_len:]
            decoded = processor.decode(generation, skip_special_tokens=True)

        # Log the successful generation of the caption
        logging.info("Caption generated successfully.")

        # Return the generated caption
        return decoded

    # Handle exceptions that may occur during caption generation
    except Exception as e:
        # Custom exception handling
        raise CustomExceptionHandling(e, sys) from e
