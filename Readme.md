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
            ├── 📁 DataCollect/
            ├── 📁 all/
            ├── 📁 Fake/
            ├── 📁 Real/
            ├── 📁 SplitData/
                ├── 📁 test/
                ├── 📁 train/
                ├── 📁 val/
        ├── 📁 Models/ # YOLO weights and model configs 
        ├── 📁 Testing Scripts/ # Python scripts for model inference and webcam testing 
        └── README.md
        └── dataCollection.py
        └── main.py
        └── splitData.py

## 🚀 How to Run

1. Clone the repo  
   ```bash
    git clone https://github.com/arnavco9573/Antispoofing.git
    cd Antispoofing
    pip install -r requirements.txt
    python main.py
   ```

## 🧾 Dataset Preparation

To prepare the dataset for training the anti-spoofing model:

1. **Create folder structure** in the root `Antispoofing/` directory as shown above:

2. **Collect data using the webcam**:
- Run your data collection script (`dataCollection.py`).
- When collecting **fake** images, make sure `classId = 0`.
- After collecting fake images, move them into the `fake/` folder.

3. Repeat the process for **real** faces:
- Set `classId = 1`.
- After collecting, move the real images into the `real/` folder.

4. Once you have both real and fake images:
- Copy all images from `fake/` and `real/` into the `all/` folder.

5. **Split the dataset** using the provided script:
- Run `splitData.py` to automatically create YOLO-compatible folders:
  ```
  📁 train/
  📁 val/
  📁 test/
  data.yaml
  ```
- These will be auto-generated with the correct format for training with YOLO.

> Make sure to check the script paths and verify images/labels are aligned correctly before training.

> For best Results take the dataset of 5000 to 6000 images and run them for atleast 100 epochs the training you can done on the google colab
## 🏋️ Training the Model

For best results, collect a **dataset of 5000 to 6000 images** and run training for at least **100 epochs**. You can train the model on **Google Colab** by following these steps:

1. **Prepare the dataset on Google Drive**:
   - Place your dataset in a folder like `Datasets/Spoofing/Version2` in your Google Drive.
   - Make sure you have the `data.yaml` file (created in the previous dataset preparation step) located in the same folder.

2. **Mount Google Drive in Google Colab**:
   ```python
    from google.colab import drive
    drive.mount('/content/gdrive')
    folder_path_drive = "Datasets/Spoofing/Version2"
    !cp /content/gdrive/MyDrive/{folder_path_drive}/data.zip /content
    ```
3. **Copy the dataset to Colab**:
    ```python
    folder_path_drive = "Datasets/Spoofing/Version2"
    !cp /content/gdrive/MyDrive/{folder_path_drive}/data.zip /content
   ```
4. **Unzip the dataset**:
    ```pyhton
    %cd /content
    !unzip -q data.zip -d /content/Data
   ```
5. **Make sure your data.yaml file is in the correct path in Google Drive. The YAML file should include information like class names and the paths to the train, val, and test directories.**

6. **Run the YOLO training command:**
    ```pyhton
    !yolo task=detect mode=train model=yolov8l.pt data=../content/Data/data.yaml epochs=300 imgsz=640 patience=25
   ```

