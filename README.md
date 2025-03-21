# Driver Drowsiness Detection using Convolutional Neural Networks (CNN) 🚗💤

This project develops a deep learning model to detect driver drowsiness using images from the "Driver Drowsiness Dataset (DDD)" on Kaggle. It aims to classify drivers as "Drowsy" or "Non Drowsy" to improve road safety by mitigating fatigue-related accidents.

## Features ✨
- **Dataset**: Leverages the "Driver Drowsiness Dataset (DDD)" with over 30,000 images of drivers in drowsy and non-drowsy states. 📊
- **Model**: A Convolutional Neural Network (CNN) built with TensorFlow/Keras, incorporating convolutional layers, max-pooling, and dense layers for robust classification. 🧠
- **Data Augmentation**: Employs techniques such as rotation, flipping, and brightness adjustments to enhance model generalization. 🔄
- **Training**: Utilizes callbacks like EarlyStopping, ModelCheckpoint, and ReduceLROnPlateau to optimize training and avoid overfitting. ⚙️
- **Evaluation**: Achieves up to 99.9% accuracy on validation data, with visually appealing plots for accuracy and loss metrics. 📈
- **Prediction**: Enables users to upload custom images for real-time drowsiness prediction with a user-friendly interface. 🖼️

## Technologies Used 🛠️
- Python
- TensorFlow/Keras
- OpenCV (cv2)
- NumPy
- Matplotlib
- Scikit-learn
- Google Colab / Kaggle Notebooks

## How to Run 🚀
1. Clone the repository:  
   ```bash
   git clone https://github.com/yourusername/driver-drowsiness-detection.git

### Dataset🚀
- Download the dataset from [Kaggle](https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd) and place it in the project directory.
- Run the Jupyter notebook or Python script to train the model and test predictions.

### Results ✅
- **Training accuracy**: ~99.9%
- **Validation accuracy**: ~99.9%
- **Loss**: As low as 0.0004 on validation data
- **Visualizations**: Stylish graphs displaying training/validation accuracy and loss. 📉

### Future Improvements 🔮
- Integrate real-time video processing for live detection.
- Enhance performance with transfer learning (e.g., ResNet or VGG).
- Deploy the model as a web or mobile application.

### License 📜
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Contributing 🤝
Contributions are welcome! Feel free to fork this repository, report issues, or submit pull requests to enhance the project.
