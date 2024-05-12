'''import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load the dataset
dataset = pd.read_csv('DigitalAd_dataset.csv')

# Separate features and target variable
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

# Feature scaling
sc = StandardScaler()
X_scaled = sc.fit_transform(X)

# Train the model
model = LogisticRegression(random_state=0)
model.fit(X_scaled, Y)

# Define the Streamlit app
def main():
    st.title('Customer Purchase Prediction')

    # Get user input for age and salary
    age = st.number_input('Enter Customer Age:', min_value=0, max_value=100, step=1)
    salary = st.number_input('Enter Customer Salary:', min_value=0, step=100)

    # Make prediction
    new_cust = np.array([[age, salary]])
    new_cust_scaled = sc.transform(new_cust)
    #prediction = model.predict(new_cust_scaled)

    # Predict probability for class 1 (buying)
    probability = model.predict_proba(sc.transform(new_cust))[:, 1]

    # Manually choose a threshold
    threshold = 0.5

    # Compare probability to threshold
    if probability > threshold:
        prediction = 1
    else:
        prediction = 0


    # Display prediction result
    st.subheader('Prediction:')
    if prediction == 1:
        st.write('Customer will Buy')
    else:
        st.write("Customer won't Buy")

if __name__ == '__main__':
    main()
'''