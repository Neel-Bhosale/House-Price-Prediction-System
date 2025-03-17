# House Price Prediction using Machine Learning:
This project builds and deploys a machine learning model to predict house prices based on various features such as location, number of rooms, population, and income levels. The model is deployed using FastAPI for real-time predictions.

## Project Overview:
### Data Preprocessing:
1. Loaded and explored the dataset.
2. Handled missing values and performed feature engineering.
3. Encoded categorical variables and scaled numerical features.

### Model Training & Evaluation:
1. Used XGBoost as the regression model.
2. Split data into training and testing sets.
3. Evaluated model performance using RMSE, MAE, and R² scores.

### Model Deployment:
1. Saved the trained model using Pickle.
2. Built a FastAPI application to serve predictions.
3. Created a /predict endpoint to take input features as JSON and return the predicted house price.
4. Successfully tested the API using Postman & Python requests.

## Project Structure
📂 House-Price-Prediction  
│── Housing.csv                 # Contains the housing dataset  
│── best_xgb_model.pkl       # Trained machine learning model  
│── app.py                   # FastAPI application for prediction  
│── house_price_prediction.ipynb  # Jupyter Notebook (EDA, training, evaluation)  
│── README.md                # Project documentation  
│── requirements.txt         # Python dependencies
 
