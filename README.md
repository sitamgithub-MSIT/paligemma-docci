# PaliGemma DOCCI

PaliGemma DOCCI is a web application that generates image captions in multiple languages, including English, French, and Spanish. The application uses a fine-tuned version of PaliGemma2, part of a family of versatile vision-language models built on the DOCCI dataset. It produces captions for the input image in the selected language.

## Project Structure

The project is structured as follows:

- `src\`: The folder that contains the source code for the project.

  - `paligemma\`: The folder containing the source code for the PaliGemma2 model loading and response generation.

    - `model.py`: The file that contains the code for loading the model and the processor.
    - `response.py`: The file that contains the function for generating the caption for the input image and language.

  - `config.py`: This file contains the configuration for the used model.
  - `logger.py`: This file contains the project's logging configuration.
  - `exception.py`: This file contains the exception handling for the project.

- `app.py`: The main file that contains the Gradio application for image captioning.
- `requirements.txt`: The file containing the project's required dependencies.
- `LICENSE`: The license file for the project.
- `README.md`: The README file that contains information about the project.
- `assets`: The folder that contains the screenshots for working on the application.
- `images`: The folder that contains the images for testing the application.

## Tech Stack

- Python (for the programming language)
- PyTorch (for the deep learning framework)
- Hugging Face Transformers Library (for the visual language model)
- Gradio (for the web application)
- Hugging Face Spaces (for hosting the gradio application)

## Getting Started

To get started with this project, follow the steps below:

1. Clone the repository: `git clone https://github.com/sitamgithub-MSIT/paligemma-docci.git`
2. Change the directory: `cd paligemma-docci`
3. Create a virtual environment: `python -m venv tutorial-env`
4. Activate the virtual environment: `tutorial-env\Scripts\activate`
5. Install the required dependencies: `pip install -r requirements.txt`
6. Run the Gradio application: `python app.py`

Now, open up your local host and see the web application running. For more information, please refer to the Gradio documentation [here](https://www.gradio.app/docs/interface). Also, a live version of the application can be found [here](https://sitammeur-paligemma-docci.hf.space/).

**Note**: You need a Hugging Face access token to run the application. You can get the token by signing up on the Hugging Face website and creating a new token from the settings page. After getting the token, you can set it as an environment variable `ACCESS_TOKEN` in your system by creating a `.env` file in the project's root directory. Check the `.env.example` file for reference.

The application is hosted on Hugging Face Spaces running on a GPU. You are expected to have a GPU for use when running the application. If you do not have a GPU, you can use the live version of the application.

## Usage

The web application allows you to input an image and select a language for the generated captions. After rendering the image and selecting the language, you can click on the "Submit" button to get the captions for the image. The application will display the generated captions for the image in the selected language.

## Contributing

Contributions are welcome! If you would like to contribute to this project, please raise an issue to discuss the changes you want to make. Once the changes are approved, you can create a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

If you have any questions or suggestions regarding the project, feel free to reach out to me on my GitHub profile.

Happy coding! 🚀
