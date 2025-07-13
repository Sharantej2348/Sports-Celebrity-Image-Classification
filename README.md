
# 🏆 Sport Celebrity Image Classification

This project demonstrates a complete machine learning pipeline for classifying images of sport celebrities using facial recognition and image processing techniques. It includes face and eye detection, wavelet-based feature extraction, model training using GridSearchCV, and making predictions on new data.

---

## 📁 Project Structure

1. **Image Preprocessing**
   - Detects faces and eyes using Haar cascades.
   - Crops images based on facial region.
   - Saves processed images for training.

2. **Feature Extraction**
   - Applies **Wavelet Transform** to enhance image features (edges, facial contours).

3. **Model Training**
   - Uses **GridSearchCV** to train multiple classifiers:
     - SVM
     - RandomForest
     - Logistic Regression
   - Selects the best performing model based on accuracy.

4. **Model Saving**
   - Trained model, class dictionary, and label encoders are saved using `joblib` for future inference.

5. **Prediction Interface**
   - Loads a new image, performs preprocessing and feature extraction.
   - Predicts the sport celebrity using the saved model.

---

## 🛠️ Technologies Used

- Python
- OpenCV
- NumPy, Pandas
- PyWavelets (Wavelet Transform)
- Scikit-learn
- Matplotlib
- Jupyter Notebook

---

## 🔍 How It Works

1. **Face & Eye Detection**: Uses Haar cascades to identify and isolate relevant facial features.
2. **Wavelet Transform**: Enhances important patterns and structures in the image.
3. **Feature Vector Creation**: Concatenates raw image and wavelet image into a single flattened array.
4. **Model Training**: Several ML models are tuned via GridSearchCV to determine the optimal parameters.
5. **Model Inference**: On unseen test data, the pipeline predicts the correct sport celebrity with high confidence.

---

## 📷 Sample Input

Example of an input image:

```python
img = cv2.imread('./test_images/sharapova1.jpg')
```

The image is then processed for face detection, cropped, and used for prediction.

---

## 🧠 Prediction Output

```
Predicted Celebrity: Maria Sharapova
```

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/celebrity-classification.git
   cd celebrity-classification
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Open the notebook:
   ```bash
   jupyter notebook image_classification.ipynb
   ```

4. Run through each cell to process, train, and predict.

---

## 🧾 Requirements

```
opencv-python
numpy
matplotlib
pywavelets
scikit-learn
joblib
```

You can generate a `requirements.txt` using:
```bash
pip freeze > requirements.txt
```

---

## 📌 Notes

- Only images with **two detected eyes** are used for training to ensure high-quality data.
- Uses **Wavelet Transform** to extract meaningful patterns for model learning.
- You can add new celebrities by collecting and placing cropped images into the dataset folder and retraining the model.

---

## 🤝 Acknowledgements

- Haar cascade XML files from OpenCV
- Sports celebrity images used for educational purposes only
