# Multi-Modal Fake News Detection (Text + Image)

This project leverages deep learning for multi-modal (text and image) fake news detection. It enables users to classify news as real or fake via an intuitive Streamlit interface. The system integrates advanced NLP, computer vision, and transfer learning techniques, with optional user authentication for secure access.

---

## Features

- **Simultaneous Detection**: Analyze both text and images for comprehensive fake news detection.
- **Real-Time Prediction**: Interactive Streamlit GUI for instant results.
- **Model Management**: Save and load trained models for reuse.
- **Data Preprocessing**:
  - Text: Tokenization, padding, and cleaning.
  - Image: Resizing and normalization.
- **Multi-Model Architecture**: Combines features from text and image models for improved accuracy.
- **Optional Authentication**: Secure user login (optional).
- **Easy Deployment**: Simple setup for local and cloud environments.

---

## Technologies Used

- Python 3.10+
- TensorFlow / Keras (Deep Learning)
- OpenCV (Image Preprocessing)
- Transfer Learning (Pretrained Models)
- Pandas & NumPy (Data Handling)
- Matplotlib & scikit-learn (Visualization & Evaluation)
- Natural Language Processing (NLP)
- Streamlit (GUI)
- Pickle (Model Serialization)

---

## Live Demo

Access the live app here:  
                         (https://dzp3nhoyhvpr6n73fafazt.streamlit.app/)

---

## Project Structure

```
multi_modal_fake_detection/
├─ src/
│  ├─ app.py                  # Main Streamlit app
│  ├─ text_model.py           # Text classification model
│  ├─ image_model.py          # Image classification model
├─ myenv/                     # Virtual environment
├─ datasets/
│  ├─ True.csv
│  ├─ Fake.csv
│  └─ Image datasets/
├─ testcse/
├─ .gitignore
├─ requirements.txt           # Python dependencies
├─ fake_news_model.h5         # Pretrained text model
├─ tokenizer.pkl              # Saved tokenizer
├─ real_fake_face_detection_model.h5 # Pretrained image model
├─ README.md
└─ LICENSE (optional)
```

---

## Quick Start

### 1. Clone the Repository

```sh
git clone https://github.com/<your-username>/multi_modal_fake_detection.git
cd multi_modal_fake_detection/src
```

### 2. Install Dependencies

```sh
pip install -r requirements.txt
```

### 3. Run the App

```sh
streamlit run app.py
```

- Open the GUI in your browser.
- Enter news text and/or upload an image.
- Click **Predict** to classify the news as Real or Fake.
- Optionally, save results or models.

---

## Notes

- **Python Version:** 3.10+ recommended
- **Dataset:** Ensure text and image samples are correctly labeled
- **GPU (Optional):** TensorFlow GPU recommended for faster training
- **Model Loading:** Use `pickle` or `load_model()` for pretrained models

---

## License

MIT License © 2025

---


Would you like a more detailed "How to Run" section with step-by-step local/server instructions and dataset examples?  
Let me know!
