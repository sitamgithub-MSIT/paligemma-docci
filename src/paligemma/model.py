# Necessary imports
import os
import sys
from dotenv import load_dotenv
from typing import Any
import torch
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor

# Local imports
from src.logger import logging
from src.exception import CustomExceptionHandling


# Load the Environment Variables from .env file
load_dotenv()

# Access token for using the model
access_token = os.environ.get("ACCESS_TOKEN")


def load_model_and_processor(model_name: str, device: str) -> Any:
    """
    Load the model and processor.

    Args:
        - model_name (str): The name of the model to load.
        - device (str): The device to load the model onto.

    Returns:
        - model: The loaded model.
        - processor: The loaded processor.
    """
    try:
        # Load the model and processor
        model = (
            PaliGemmaForConditionalGeneration.from_pretrained(
                model_name, torch_dtype=torch.bfloat16, token=access_token
            )
            .eval()
            .to(device)
        )
        processor = PaliGemmaProcessor.from_pretrained(model_name, token=access_token)

        # Log the successful loading of the model and processor
        logging.info("Model and processor loaded successfully.")

        # Return the model and processor
        return model, processor

    # Handle exceptions that may occur during model and processor loading
    except Exception as e:
        # Custom exception handling
        raise CustomExceptionHandling(e, sys) from e
