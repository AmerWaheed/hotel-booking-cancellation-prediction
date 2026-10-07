# 🏨 Hotel Booking Cancellation Prediction

A machine learning classification project that predicts whether a hotel booking will be **canceled or not canceled** using customer, booking, and reservation-related features.

The project follows an end-to-end machine learning workflow, including **data cleaning, exploratory data analysis, feature engineering, preprocessing, model comparison, evaluation, and model persistence**.

---

## 📌 Project Overview

Hotel booking cancellations can negatively affect hotel revenue, room availability, and operational planning.

The goal of this project is to build a machine learning model that can identify bookings with a high probability of cancellation and provide useful insights into the factors associated with cancellation behavior.

### Business Objective

The model can help hotels:

* Identify bookings that are more likely to be canceled
* Improve room and revenue planning
* Understand customer booking behavior
* Support data-driven cancellation management

---

## 📊 Dataset

The project uses the **Hotel Booking Demand** dataset.

The dataset contains **119,390 hotel bookings** with information related to:

* Hotel type
* Arrival date
* Lead time
* Length of stay
* Number of guests
* Meal type
* Country
* Market segment
* Distribution channel
* Room type
* Deposit type
* Customer type
* Previous bookings
* Special requests
* Average Daily Rate (ADR)
* Cancellation status

### Target Variable

`is_canceled`

* `0` → Not Canceled
* `1` → Canceled

---

## 🔍 Data Quality & Preprocessing

The dataset was inspected for:

* Missing values
* Duplicate records
* Data types
* Numerical and categorical variables
* Outliers
* Target distribution

Missing values were identified in:

* `children`
* `country`
* `agent`
* `company`

Duplicate records were also checked.

### Data Leakage Prevention

Two columns were removed because they contain information that would only be known after the booking outcome:

```text
reservation_status
reservation_status_date
```

Including these features would cause **data leakage** because they are directly related to the final reservation status.

The following columns were also removed because they were not useful for generalization:

```text
name
email
phone-number
credit_card
agent
company
```

---

## 🛠️ Feature Engineering

Several new features were created to capture more meaningful booking behavior.

### Engineered Features

* `total_guests`
* `total_nights`
* `total_cost`
* `price_per_guest`
* `booking_per_night`
* `room_changed`
* `family`
* `special_request_ratio`

These features provide the models with additional information about booking size, cost, stay duration, room changes, family status, and customer requests.

---

## 📈 Exploratory Data Analysis

The analysis investigated relationships between booking characteristics and cancellation behavior.

Key areas explored included:

* Target distribution
* Hotel type
* Market segment
* Deposit type
* Customer type
* Lead time
* ADR distribution
* Monthly cancellation trends
* Country
* Reserved vs. assigned room type
* Special requests
* Total booking cost

### Key Finding

Bookings with **longer lead times showed a noticeably higher cancellation rate**, indicating that how far in advance a booking is made can be an important factor in cancellation behavior.

---

## 🤖 Machine Learning

The following classification models were trained and compared:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. XGBoost

A preprocessing pipeline was used to handle numerical and categorical features.

Categorical variables were encoded using **One-Hot Encoding**, while numerical features were processed separately.

---

## 📊 Model Performance

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score

Because the project focuses on cancellation prediction and the dataset contains some class imbalance, **F1 Score** was given particular attention.

| Model               |  Accuracy | Precision |    Recall |  F1 Score |
| ------------------- | --------: | --------: | --------: | --------: |
| Logistic Regression |     0.820 |     0.810 |     0.672 |     0.735 |
| Decision Tree       |     0.854 |     0.803 |     0.802 |     0.803 |
| **Random Forest**   | **0.890** | **0.889** | **0.802** | **0.843** |
| XGBoost             |     0.873 |     0.853 |     0.795 |     0.823 |

### 🏆 Best Model: Random Forest

The Random Forest model achieved the best F1 Score among the evaluated models:

**F1 Score: 0.843**

It also achieved:

* **Accuracy:** 88.97%
* **Precision:** 88.93%
* **Recall:** 80.20%

### Classification Report

For the final Random Forest model:

| Class        | Precision | Recall |   F1 |
| ------------ | --------: | -----: | ---: |
| Not Canceled |      0.89 |   0.94 | 0.91 |
| Canceled     |      0.89 |   0.80 | 0.84 |

---

## 🧠 Machine Learning Pipeline

The overall workflow was:

```text
Raw Dataset
     ↓
Data Understanding
     ↓
Data Quality Check
     ↓
Remove Data Leakage
     ↓
Feature Engineering
     ↓
Exploratory Data Analysis
     ↓
Train / Test Split
     ↓
Preprocessing Pipeline
     ↓
Model Training
     ↓
Model Comparison
     ↓
Evaluation
     ↓
Final Random Forest Model
     ↓
Model Persistence
```

---

## 💻 Technologies Used

### Programming

* Python

### Data Analysis

* Pandas
* NumPy

### Visualization

* Matplotlib
* Seaborn
* Plotly

### Machine Learning

* Scikit-learn
* XGBoost

### Model Persistence

* Joblib

### Main ML Techniques

* Classification
* Feature Engineering
* One-Hot Encoding
* Train/Test Split
* Random Forest
* Decision Tree
* Logistic Regression
* XGBoost
* Model Evaluation

---

## 📁 Project Structure

```text
hotel-booking-cancellation-prediction/
│
├── Hotel_Booking_cancellation_prediction.ipynb
├── hotel_booking_model.pkl
├── countries.pkl
├── README.md
└── images/
    ├── target_distribution.png
    ├── correlation_heatmap.png
    ├── model_comparison.png
    └── confusion_matrix.png
```

---

## 🚀 Future Improvements

Possible improvements include:

* More extensive hyperparameter optimization
* Advanced threshold optimization
* Cross-validation
* Feature importance analysis
* SHAP explainability
* Handling class imbalance more explicitly
* Deployment as a web application
* Real-time cancellation probability prediction
* Monitoring model performance after deployment

---

## 🎯 Conclusion

This project demonstrates an end-to-end machine learning workflow for a real-world business problem.

The final Random Forest model achieved an **88.97% accuracy and 0.843 F1 Score**, outperforming Logistic Regression, Decision Tree, and XGBoost on the evaluated test set.

The project also demonstrates the importance of **data leakage prevention, feature engineering, preprocessing pipelines, model comparison, and business-oriented evaluation** when developing a machine learning solution.

---

## 👤 Author

**Amer Waheed**

Computer Science Student | Data Analyst & ML Enthusiast

**Skills:** Python • SQL • Power BI • Excel • Machine Learning • Deep Learning
