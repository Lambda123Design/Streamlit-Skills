# Streamlit-Skills

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
