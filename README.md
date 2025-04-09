# DigitNet

DigitNet is a web application for handwritten digit recognition powered by a machine learning model.

## Features

- Upload images and get digit predictions
- View prediction results on the web interface
- Train your own digit recognition model with the provided script

## Tech Stack

- Python
- Flask
- Machine Learning (likely using libraries such as TensorFlow or PyTorch)

## Getting Started

### Installation

1. Clone the repository
2. Install dependencies:

```
pip install -r requirements.txt
```

### Running the App

```
python app.py
```

The app will start a local server. Open your browser and navigate to `http://localhost:5000`.

### Training the Model

To train or retrain the digit recognition model, run:

```
python train_model.py
```

## Project Structure

- `app.py` - Flask web application
- `train_model.py` - Script to train the ML model
- `requirements.txt` - Python dependencies
- `templates/` - HTML templates
- `static/` - Static files (images, CSS, JS)

## License

This project is licensed under the MIT License.
