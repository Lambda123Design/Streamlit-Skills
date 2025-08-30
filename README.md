# Streamlit-Skills

A) Basic Working on Streamlit

B) Steps for Deployment on Streamlit Cloud:

C) Development of Streamlit App for Simple RNN Project - IMDb Sentiment Analysis

D) Development of Streamlit App for LSTM Next Word Prediction

**A) Basic Working on Streamlit**

1. Open a app.py in VS Code, and code st.title("Hello Streamlit")

2. Run in command prompt using "streamlit run app.py"

3. Taking the same to next level: Display a Simple Text: st.write("This is a simple text")

4. Display a df: st.write(df)

5. **Create a Line Chart**

chart_data=pd.DataFrame(np.random.randn(20,3), columns=['a','b','c'])

st.line_chart()

**Execute the same way in Command Prompt**

6. Creating the Widgets:

(i) Created a file called: widgets.py

(ii) st.title("Streamlit Text Input")

(iii) names=st.text_input("Enter your name:")

(iv) if name: st.write(f"Hello, {name:}")

(v) Run in Command Prompt using, "streamlit run widgets.py"

(vi) Creating a slider: st.slider("Select your age:",0,100,25)

(vii) st.write(f"Your age is {age}.")

**We can also use it for SelectBox**

(viii) We can also upload files:

uploaded_file=st.file_uploader("Choose a CSV File",type="csv")

if uploaded_file is not None:
   df=pd.read_csv(uploaded_file); st.write(df)

## B) Steps for Deployment on Streamlit Cloud:

(i) Take the pickle file along with the requirement.txt; you can commit via GitHub command line or directly copy–paste files. Upload all files (Readme, app.py, pickle file, requirement.txt).

(ii) Reload to process files; confirm all are visible in GitHub repository.

(iii) Go to share.streamlit.io → click Create app. Choose GitHub repository, paste repo URL, set branch as main, and file path as app.py.

(iv) Streamlit creates an app URL; deploy it step by step.

(v) Optionally add environment variables or secret keys using the advanced token section (encrypted and secure).

(vi) On deployment, Streamlit installs packages from requirement.txt, creates environment, and sets up the platform (takes ~30–45 seconds).

(vii) During this time it runs pip install -r requirement.txt; wait until installation finishes.

(viii) Status shows “app is in the oven / warming up” — meaning server is preparing.

(ix) Once ready, the app runs on the generated URL, accessible publicly in the Streamlit cloud.

### C) Development of Streamlit App for Simple RNN Project - IMDb Sentiment Analysis:

(i) Create a main.py file to combine all functionality from prediction.ipynb (decoding reviews, preprocessing text, loading trained model).

(ii) Import necessary libraries:

(a) "import streamlit as st"

(b) Load trained model with load_model

(c) Import IMDb dataset, sequence preprocessing utilities

(d) Load word index for converting reviews to vector indices

(e) Define helper functions for decoding reviews and preprocessing text

(iii) Define the prediction function

(iv) Set up Streamlit interface: 

(a) Add title and description: "st.title("IMDb Movie Review Sentiment Analysis")"; "st.write("Enter a movie review to classify it as positive or negative.")"

(b) Create user input area: "user_input = st.text_area("Enter your movie review:")"

(c) Add button for prediction: "if st.button("Classify"):
    processed_input = preprocess_text(user_input)
    prediction = model.predict(processed_input)
    sentiment = "positive" if prediction[0][0] > 0.5 else "negative"
    st.write(f"Sentiment: {sentiment}")
    st.write(f"Prediction score: {prediction[0][0]}")
else:
    st.write("Please enter a movie review.")" 

(d) Optionally, display a sample review using decode_review helper function)

(v) Run the app:

(a) Open terminal → navigate to folder containing "main.py"

(b) Execute: "streamlit run main.py"

(c) App launches in browser; enter a review and click Classify

(d) Model outputs sentiment (positive/negative) and prediction score (confidence)

(vi) Deployment (Krish Gave as a Assignment, as it was explained in ANN Project Video:)

(a) Upload all files to GitHub

(b) Deploy directly on Streamlit Cloud

This completes the Streamlit app workflow, from input interface, preprocessing, prediction, displaying results, to deployment.

**D) Development of Streamlit App for LSTM Next Word Prediction**

1. Create app file: Inside LSTM_RNN folder, create app.py for the Streamlit web interface and prediction logic.

2. Import required libraries:

"import pickle
import numpy as np
import streamlit as st
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences"

3. Load model and tokenizer:

"model = load_model("next_word_LSTM.h5")

with open("tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)"

4. Add prediction function (reuse from notebook): Takes model, tokenizer, input text, and max sequence length → preprocesses, pads sequence, and predicts next word.

5. Build Streamlit UI:

(i) App title: "st.title("Next Word Prediction with LSTM and Early Stopping")"

(ii) Text input box (with default text): "input_text = st.text_input("Enter a sequence of words:", "to be or not to be")"

(iii) Prediction button:

"if st.button("Predict Next Word"):
    max_sequence_len = model.input_shape[1] + 1
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
    st.write(f"Predicted next word: {next_word}")"

6. Run the app:

(i) Navigate to project folder in terminal

(ii) Run: "streamlit run app.py"

7. App behavior:

(i) Loads model and tokenizer into memory (takes a little time)

(ii) UI shows title, input box, and button

(iii) Enter a phrase → click Predict Next Word → app displays predicted next word

8. Example inputs & outputs:

(i) “to be or not to be” → “considered” / “crack”

(ii) “Well, good night if you do meet” → “two”

(iii) “Welcome” → “friends”

(iv) “Tis but our” → “fantasy”

9. Users can try different inputs; performance improves with more training epochs.
