Multi-Modal Fake News Detection (Text + Image):

A collaborative AI-based tool to detect fake news in text and images. This project combines deep learning models for text and image analysis, allowing users to classify news as real or fake. Users can also input their own datasets, save predictions, and interact via a simple GUI.

Originally inspired by modern AI research, extended with multi-modal deep learning, GUI interface, and optional user authentication.

Features:

Detect fake news from text and images simultaneously

Real-time prediction via GUI

Save and load trained models

Clean preprocessing of text (tokenization, padding, cleaning)

Image preprocessing (resizing, normalization)

Multi-model architecture combining text and image features

Optional: user authentication for secure access

Interactive Streamlit interface

Technologies Used

Python 3.10+

TensorFlow / Keras (deep learning)

OpenCV (image preprocessing)

Transfer learning with pretrained models

Pandas & NumPy (data handling)

Matplotlib & scikit-learn

NLP (natural language processing)

Streamlit (GUI)

Pickle (saving/loading models)

Live Link : https://dzp3nhoyhvpr6n73fafazt.streamlit.app/



Project Structure:  

multi_modal_fake_detection/
├─ src/
│  ├─ app.py              # Main Streamlit app
│  ├─ text_model.py       # Text classification model
│  ├─ image_model.py      # Image classification model
├─ myenv                  # Virtual environment
├─ datasets/ 
│    ├─ True.csv 
│    ├─ Fake.csv
│    └─ Image datasets
├─ testcse
├─ .gitignore
├─ requirements.txt       # Required Python packages
├─ fake_news_model.h5
├─ tokenizer.pkl
├─ real_fake_face_detection_model.h5
├─ README.md
└─ LICENSE (optional)

Quick Start
1. Clone the Repository
git clone https://github.com/<your-username>/multi_modal_fake_detection.git
cd multi_modal_fake_detection/src

2. Install Dependencies
pip install -r requirements.txt

3. Run the App
streamlit run app.py


Open the GUI in your browser

Input text and/or upload an image

Click Predict to classify news as Real or Fake

Optional: Save results or models

Notes

Python Version: 3.10+ recommended

Dataset: Ensure text and image data are properly labeled

GPU (optional): TensorFlow GPU recommended for faster training

Model Loading: Pretrained models can be loaded with pickle or load_model()

License

MIT License © 2025

If you want, I can also add a detailed “How to Run” section like the BeatBox README, including examples for running locally, with datasets, or deploying on a server, so your README looks professional and step-by-step friendly.

Do you want me to do that next?
