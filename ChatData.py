import streamlit as st  # Import Streamlit for creating the web app
import pandas as pd  # Import pandas for data handling
from langchain.chat_models import ChatOpenAI  # Correct import for ChatOpenAI from LangChain
from langchain.prompts import ChatPromptTemplate  # Import ChatPromptTemplate for formatting chat prompts
from langchain.chains import LLMChain  # Import LLMChain to create a chain for processing input

# Set up the title and description for the Streamlit app
st.title("LangChain Chatbot & Data Analyzer")
st.write("Ask any question and get a response from the GPT-4 model. You can also upload a CSV file for basic data analysis.")

# Streamlit input field for API key
api_key = st.text_input("Enter your OpenAI API key:", type="password")

# Only create an instance of ChatOpenAI if an API key is provided
if api_key:
    # Create an instance of ChatOpenAI LLM with the provided API key and model name
    llm = ChatOpenAI(openai_api_key=api_key, model="gpt-4")

    # Define a dynamic chat prompt template with a placeholder for user queries
    prompt_template = ChatPromptTemplate.from_messages([("user", "{user_query}")])

    # Create an LLMChain using the LLM and the chat prompt template
    chain = LLMChain(llm=llm, prompt=prompt_template)

    # Function to get response from the query
    def get_response_from_query(query):
        # Run the chain with the input dictionary specifying the placeholder values
        response = chain.run({"user_query": query})
        return response

    # Streamlit input field for user query
    user_query = st.text_input("Enter your question:")

    # If the user inputs a query, get and display the response
    if user_query:
        with st.spinner("Generating response..."):
            response = get_response_from_query(user_query)
            st.write("Response:", response)

else:
    st.warning("Please enter your OpenAI API key to use the chatbot.")

# File uploader for CSV files
uploaded_file = st.file_uploader("Upload a CSV file for data analysis", type=["csv"])

# If a file is uploaded, read it using pandas and display the data
if uploaded_file:
    df = pd.read_csv(uploaded_file)  # Read CSV file
    st.write("Uploaded CSV Data:")
    st.dataframe(df)  # Display the DataFrame

    # Display basic data analysis options
    st.write("Basic Data Analysis:")

    # Display column names
    if st.checkbox("Show column names"):
        st.write(df.columns.tolist())

    # Display summary statistics
    if st.checkbox("Show summary statistics"):
        st.write(df.describe())

    # Show data types
    if st.checkbox("Show data types"):
        st.write(df.dtypes)

    # Option to show the first n rows
    num_rows = st.slider("Select number of rows to view", min_value=1, max_value=len(df))
    st.write(df.head(num_rows))
