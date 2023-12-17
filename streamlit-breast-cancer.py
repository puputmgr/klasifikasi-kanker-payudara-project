# Import library
import streamlit as st
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.DataFrame(data.target, columns=['target'])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a random forest classifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Streamlit App
st.title('Breast Cancer Detection App')

# Sidebar
st.sidebar.header('User Input Features')

# Display the user input features
def user_input_features():
    input_features = {}
    for feature in X.columns:
        input_features[feature] = st.sidebar.slider(f'{feature} value', float(X[feature].min()), float(X[feature].max()), float(X[feature].mean()))
    return pd.DataFrame(input_features, index=[0])

user_input = user_input_features()

# Display the user input
st.subheader('User Input:')
st.write(user_input)

# Make predictions
prediction = clf.predict(user_input)

# Display the prediction
st.subheader('Prediction:')
st.write('Class:', prediction[0])
st.write('Prediction Probability:', clf.predict_proba(user_input)[0])

# Evaluate model performance
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.subheader('Model Performance:')
st.write(f'Accuracy: {accuracy:.2f}')
st.write('Classification Report:')
st.write(classification_report(y_test, y_pred))

