# Psyliq-Data-Scientist-Intern-Projects


## 1. **Stock Price Prediction**

### Description
The project predicts stock prices using historical data with the help of LSTM (Long Short-Term Memory) neural networks. The model predicts future prices and provides suggestions for intraday or swing trading based on user inputs.

### Features
- Predicts future stock prices using LSTM.
- Provides trading suggestions based on user preferences (buy/sell, intraday/swing trading).
- Normalizes stock price data for efficient model training.

### Tech Stack
- Python
- TensorFlow/Keras
- Pandas, NumPy, Scikit-learn
- Matplotlib for visualization

### How to Run
1. Install required Python libraries: `pip install tensorflow pandas numpy scikit-learn matplotlib`.
2. Load stock price data in `stock_data.csv` with `Date` and `Close` columns.
3. Run the Python script.
4. Input stock details and trading preferences to get predictions and suggestions.
### Output
![image](https://github.com/user-attachments/assets/20a9944d-5a82-4099-8557-06acbb9ee354)


---

## 2. **Titanic Survival Prediction**

### Description
The project predicts passenger survival on the Titanic using machine learning. It utilizes the Titanic dataset for classification based on passenger information.

### Features
- Handles missing values and preprocesses categorical data.
- Trains a Random Forest Classifier to predict survival.
- Evaluates the model's performance using accuracy and classification reports.

### Tech Stack
- Python
- Scikit-learn
- Pandas, NumPy
- Matplotlib for visualization

### How to Run
1. Install required Python libraries: `pip install pandas numpy scikit-learn`.
2. Place the Titanic dataset (`titanictrain.csv`) in the same directory as the script.
3. Run the Python script to train the model and view the results.

### Output
![image](https://github.com/user-attachments/assets/bc86b5f9-64b1-4f29-a138-04d2ff4200f5)

---

## 3. **Handwritten Digit Recognition**

### Description
This project recognizes handwritten digits using a neural network and provides a graphical interface for digit input. The model is trained on the MNIST dataset.

### Features
- Trains a neural network to recognize digits from 0 to 9.
- Provides a Tkinter-based GUI for users to draw digits.
- Displays predictions for drawn digits in real-time.

### Tech Stack
- Python
- TensorFlow/Keras
- Pandas, NumPy
- Tkinter for GUI

### How to Run
1. Install required Python libraries: `pip install tensorflow pandas numpy pillow`.
2. Place the training and test datasets (`digittrain.csv`, `digittest.csv`) in the same directory.
3. Run the Python script to train the model and launch the GUI.
4. Draw a digit on the canvas and click the "Predict" button to see the prediction.

### Output
![image](https://github.com/user-attachments/assets/dec45fee-c8ac-4ee4-b6a1-51e78386f2e2)

---

## General Requirements
- Python 3.7 or above
- Necessary datasets (provided in the respective project directories)

## License
This repository is licensed under the MIT License. See `LICENSE` for more details.

---

Feel free to use or contribute to these projects!
