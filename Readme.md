Multi-Modal Fake News Detection (Text + Image):

A collaborative AI-based tool to detect fake news in text and images. This project combines deep learning models for text and image analysis, allowing users to classify news as real or fake. Users can also input their own datasets, save predictions, and interact via a simple GUI.


Features:

Detect fake news from text and images simultaneously

Real-time prediction via GUI

Save and load trained models

Clean preprocessing of text (tokenization, padding, cleaning)

Image preprocessing (resizing, normalization)

Multi-model architecture combining text and image features

Optional: user authentication for secure access

Interactive Streamlit interface


Technologies Used:


Python 

TensorFlow / Keras (deep learning)

OpenCV (image preprocessing)

Transfer learning with pretrained model

Pandas & Numpy (data handling)

Matplotlib & scikit-learn

NLP(natural language processing)

Streamlit (GUI)

Pickle (saving/loading models)



Project Structure:

multi_modal_fake_detection/
├─ src/
│  ├─ app.py              
│  ├─ text_model.py       
│  ├─ image_model.py      
│      
│- myenv              
├─ datasets/ 
|    |- True.csv 
|    |- Fake.csv
|    |-Image datasets
|
|- testcse
|- .gitignore           
├─ requirements.txt       # Required Python packages
|- fake_news_model.h5
|- tokenizer.pkl
|- real_fake_face_detection_model.h5
├─ README.md
└─ LICENSE (optional)


Live link:
          

Quick Start :

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

Notes:

Python Version: Use Python 3.10+

Dataset: Ensure text and image data are properly labeled

GPU (optional): TensorFlow GPU recommended for faster training

Model Loading: Pretrained models can be loaded with pickle or load_model()


License

MIT License © 2025
