Smart Sorting: Transfer Learning For Identifying Rotten Fruits And Vegetables
=============================================================================

Project Overview
----------------
This is a web application that uses deep learning (Transfer Learning via MobileNetV2) to classify images of fruits and vegetables as Healthy or Rotten.

Setup Instructions
------------------
1. Install Python 3.8+ 
2. Install dependencies:
   pip install -r requirements.txt
3. Dataset Setup:
   - Create a folder named `dataset` in this directory.
   - For training, organize your images like so:
     dataset/train/healthy/
     dataset/train/rotten/
     dataset/test/healthy/
     dataset/test/rotten/
4. Training the Model:
   - Run `python train_model.py` to train the model over your dataset.
   - Wait for `healthy_vs_rotten.h5` to be generated.
5. Running the Application:
   - Run `python app.py` to start the Flask server.
   - Open your browser and navigate to `http://localhost:5000` to interact with the UI.

Note:
If you run `app.py` without training a model first, it will run in "dummy mode" returning random predictions for UI testing purposes.
