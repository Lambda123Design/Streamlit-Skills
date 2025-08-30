# Streamlit-Skills

A) Basic Working on Streamlit

B) Steps for Deployment on Streamlit Cloud:

C) Development of Streamlit App for Simple RNN Project - IMDb Sentiment Analysis

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


