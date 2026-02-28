#  Emotional Classification in Speech Processing

##  Overview
This project focuses on **Speech Emotion Recognition (SER)** using Deep Learning and Transformer-based models.  
The system analyzes human speech audio signals and automatically classifies emotions such as **happy, sad, angry, fearful, neutral**, and more.

The model leverages **Facebook's Wav2Vec2** architecture for high-accuracy emotion detection from raw audio signals.

---

##  Features
-  Emotion recognition from speech audio
-  Transformer-based Deep Learning model (Wav2Vec2)
-  Automated dataset preprocessing
-  End-to-end training pipeline
-  Evaluation using Accuracy, Precision, Recall & F1-score
-  Scalable architecture for real-world applications

---

##  Emotion Classes
The model classifies speech into the following emotions:

- Neutral  
- Calm  
- Happy  
- Sad  
- Angry  
- Fearful  
- Disgust  
- Surprised  

---

##  Project Structure
```text
Emotional-Classification-in-Speech-Processing/
│
├── app.py # Application interface/inference
├── python train.py # Model training pipeline
├── demo video.mkv # Project demonstration
└── README.md # Project documentation
```

---

##  Technologies Used
- Python
- PyTorch
- Hugging Face Transformers
- Wav2Vec2
- Librosa
- NumPy
- Pandas
- Scikit-learn

---

##  Dataset
Emotion labels are extracted automatically from audio filenames.

Example:
``` text
03-01-05-01-02-01-12.wav
↑
Emotion Code
```

---

##  Model Architecture
- Pretrained Model: facebook/wav2vec2-base
- Sampling Rate: 16kHz
- Transformer-based Speech Encoder
- Fine-tuned for Emotion Classification

##  Installation

### Clone Repository
```bash
git clone https://github.com/your-username/Emotional-Classification-in-Speech-Processing.git
cd Emotional-Classification-in-Speech-Processing

---
``` text Install Dependencies
pip install torch transformers librosa numpy pandas scikit-learn
```
``` text  Training the Model

Update dataset path in:
load_dataset('input_file_path')
Run training:
python "python train.py"
```
 Evaluation Metrics

Model performance is evaluated using:

Accuracy

Precision

Recall

Demo

Project demo available in:https://drive.google.com/file/d/1Z4-N_trjdmD3xqCqRaGH7TY2FNkKLqtC/view?usp=sharing


