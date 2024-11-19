# DeepFake Recognition

This project is a web application for recognizing DeepFake images using a deep learning model. The application is built with Streamlit and uses a pre-trained MobileNETV2 MODEL from PyTorch.

## Table of Contents
- Installation
- Usage
- Model
- Contributing
- License

## Installation

To run this project locally, follow these steps:

1. Clone the repository:
   ```bash
   gh repo clone VladQss/final.-DEEPFAKE-recognition
   cd deepfake — копия
   
2. Create and activate a virtual environment:
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3. Install the required dependencies:
pip install -r requirements.txt

## Usage
To start the Streamlit app, run the following command:

streamlit run app.py

Upload an image of a traffic sign, and the app will display the predicted class of the sign.

## Model
The model used in this project is a MobileNETV2 MODEL pre-trained on ImageNet and fine-tuned for deepfake detection. The model is saved in the model.pth file and loaded in the Streamlit app.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.
