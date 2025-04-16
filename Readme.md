# Anti-Spoofing Face Detection System

This project is a real-time face anti-spoofing system that uses computer vision and deep learning to detect whether a face in front of a webcam is real or fake.

## 🔍 Features

- Real-time webcam-based spoof detection
- Trained on custom dataset with real and spoofed face images
- Blur filtering and preprocessing pipeline
- YOLO-based model for accurate and fast detection
- Built using Python, OpenCV, PyTorch

## 🛠 Tech Stack

- Python
- OpenCV
- PyTorch
- YOLO
- Google Colab (for training)
- PyCharm (for development)

## 📁 Project Structure 
    📂 Antispoofing/ 
        ├── 📁 Datasets/ # Raw and processed images for training/testing 
        ├── 📁 Models/ # YOLO weights and model configs 
        ├── 📁 Testing Scripts/ # Python scripts for model inference and webcam testing 
        └── README.md
        └── dataCollection.py
        └── main.py
        └── splitData.py

## 🚀 How to Run

1. Clone the repo  
   ```bash
   git clone https://github.com/yourusername/anti-spoofing-face-detection.git
   cd anti-spoofing-face-detection

    pip install -r requirements.txt
   
    python main.py

