# Streamlit-Skills

A) Basic Working on Streamlit

B) Steps for Deployment on Streamlit Cloud:

C) Development of Streamlit App for Simple RNN Project - IMDb Sentiment Analysis

D) Development of Streamlit App for LSTM Next Word Prediction

E) Example Of ML App With Streamlit Web App

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

**E) Example Of ML App With Streamlit Web App**

Streamlit makes it incredibly easy to turn a machine learning project into a web application. In this example, we are solving a classification problem using the Iris dataset. Even if you’re not fully familiar with machine learning yet, this shows how quickly you can build an interactive app to make predictions.

First, we import the necessary libraries:

import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier


If scikit-learn isn’t installed yet, you can install it using pip in the terminal:

pip install scikit-learn


Next, we load the Iris dataset and convert it into a Pandas DataFrame. This dataset has four important features: petal length, sepal length, petal width, and sepal width. We also separate the target variable, which is the species of the flower. To make this efficient, we can cache the data so it doesn’t reload every time:

@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    target = pd.DataFrame(iris.target, columns=['species'])
    return df, target

df, target = load_data()


Once the data is ready, we define the machine learning model. Here, we are using a Random Forest Classifier, which is simple to train for a classification task. We fit the model using the independent features (X) and the target variable (y):

model = RandomForestClassifier()
model.fit(df.iloc[:, :], target)


To make the app interactive, Streamlit provides sliders. Sliders allow the user to input values for sepal length, sepal width, petal length, and petal width. These values are then used as input to the trained model:

sepal_length = st.slider('Sepal Length', float(df['sepal length (cm)'].min()), float(df['sepal length (cm)'].max()))
sepal_width = st.slider('Sepal Width', float(df['sepal width (cm)'].min()), float(df['sepal width (cm)'].max()))
petal_length = st.slider('Petal Length', float(df['petal length (cm)'].min()), float(df['petal length (cm)'].max()))
petal_width = st.slider('Petal Width', float(df['petal width (cm)'].min()), float(df['petal width (cm)'].max()))

input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
prediction = model.predict(input_data)
st.write('Predicted species:', prediction[0])


Once you run the app with the command:

streamlit run classification.py


You’ll see a browser interface with sliders for all four features. As you adjust the sliders, the prediction updates in real time, showing whether the flower is setosa, versicolor, or virginica. This is a great way to visualize how input features affect model predictions without writing any HTML or frontend code.

The power of Streamlit lies in its simplicity. You don’t need to create HTML forms or handle JavaScript; everything is done through Python. Features like st.cache_data prevent reloading the dataset repeatedly, sliders provide interactive inputs, and st.write allows immediate display of predictions.

In short, with just a few lines of Python, you can create a fully interactive machine learning web app that lets users experiment with different inputs and instantly see predictions. This example with the Iris dataset is simple, but the same concept can be applied to more complex machine learning projects.

**Summary:**

1. Import Necessary Libraries

Streamlit for the web interface: import streamlit as st

Pandas for data handling: import pandas as pd

Scikit-learn for ML: from sklearn.datasets import load_iris and from sklearn.ensemble import RandomForestClassifier

Install scikit-learn if not already installed:

pip install scikit-learn

2. Load and Prepare the Iris Dataset

Use load_iris() from scikit-learn.

Convert the dataset to a Pandas DataFrame for easier handling.

Separate the features (X) and target (y):

@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    target = pd.DataFrame(iris.target, columns=['species'])
    return df, target

df, target = load_data()


@st.cache_data ensures the dataset is cached so it doesn’t reload on every interaction.

3. Train a Machine Learning Model

Use RandomForestClassifier for classification.

Fit the model using all features and the target:

model = RandomForestClassifier()
model.fit(df.iloc[:, :], target)

4. Create Interactive Inputs with Sliders

Streamlit sliders allow the user to input feature values dynamically:

sepal_length = st.slider('Sepal Length', float(df['sepal length (cm)'].min()), float(df['sepal length (cm)'].max()))
sepal_width = st.slider('Sepal Width', float(df['sepal width (cm)'].min()), float(df['sepal width (cm)'].max()))
petal_length = st.slider('Petal Length', float(df['petal length (cm)'].min()), float(df['petal length (cm)'].max()))
petal_width = st.slider('Petal Width', float(df['petal width (cm)'].min()), float(df['petal width (cm)'].max()))


Combine slider inputs into a single input for prediction:

input_data = [[sepal_length, sepal_width, petal_length, petal_width]]

5. Make Predictions

Use the trained model to predict species based on slider inputs:

prediction = model.predict(input_data)
st.write('Predicted species:', prediction[0])


Streamlit automatically updates predictions in real time as sliders are adjusted.

6. Run the Streamlit App

Run the app using the terminal:

streamlit run classification.py


This opens a browser interface with sliders for all four features, showing live predictions.

7. Key Features of Streamlit

No frontend coding required: Everything is done in Python.

Interactive sliders for user input.

Real-time updates of predictions using st.write.

Caching with @st.cache_data prevents unnecessary reloading of datasets.

8. Summary of Workflow

(i) Load the Iris dataset and prepare features and target.

(ii) Train a Random Forest Classifier on the data.

(iii) Create sliders for interactive input of feature values.

(iv) Predict the species based on slider input and display it instantly.

(v) Streamlit handles web interface, interactivity, and real-time updates automatically.
