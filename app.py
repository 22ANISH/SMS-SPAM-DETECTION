import streamlit as st
import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Load the vectorizer and the model
vectorizer = joblib.load('vectorizer.pkl')
model = joblib.load('model.pkl')

# Define a function to preprocess the input message
def preprocess_message(message):
    #CountVectorizer class takes a list of strings and transforms them into a matrix of token counts
    #This matrix can then be used by the machine learning model to make a prediction.
    return vectorizer.transform([message])

# Define a function to make a prediction
def make_prediction(message):
    #making call to preprocessing function and storing matrix in mess_count
    message_count = preprocess_message(message)
    #This method makes a prediction based on the input matrix and returns an array of one or more predictions.
    prediction = model.predict(message_count)
    #return only first element and make the predection
    return prediction[0]

# Define the Streamlit app
def main():
  
    # Set the title of the app
    st.title('SMS-Spam Detector')

    # Add a text input for the user to enter a message
    message = st.text_area('ENTER A MESSAGE:-')

    # Add a predict button
    if st.button('Predict'):
        # Make a prediction
        prediction = make_prediction(message)

        # Display the prediction
        if prediction == 1:
            st.write('THIS MESSAGE IS SPAM ❌.')
            st.image('SPAM.png',width=400)
        else:
            st.write('THIS MESSAGE IS NOT SPAM ✅.')
            st.image('ham.jpg',width=400)

# Run the app
if __name__ == '__main__':
    main()