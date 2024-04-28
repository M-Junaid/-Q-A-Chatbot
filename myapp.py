# Import necessary libraries
from langchain_openai import ChatOpenAI  # Access OpenAI's ChatGPT functionality (if needed)
from langchain_core.prompts import ChatPromptTemplate  # Build chat prompts
from langchain_core.output_parsers import StrOutputParser  # Parse output as text
from langchain_community.llms import HuggingFaceEndpoint  # Interact with Hugging Face models
from decouple import config  # Access environment variables securely

# Load API key from environment variable (avoid storing directly in code)
HUGGINGFACEHUB_API_TOKEN = config("Huggingfacehub_api_key")

# Import Streamlit for web app development
import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables from a .env file 
load_dotenv()

# Enable Langchain tracing for debugging
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Set Langchain API key from environment variable
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Define the chat prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the user queries"),
    ("user", "Question:{question}")  # User will provide the question here
])

# Set the title of the Streamlit app
st.title('Langchain Demo With Opensource Model')

# Create a text input field for users to enter their search topic
input_text = st.text_input("Search the topic you want")

# Specify the Hugging Face model repository ID
repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

# Create a Hugging Face Endpoint object with API token for model access
llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN
)

# Define the output parser to handle model responses
output_parser = StrOutputParser()

# Combine the prompt, model, and parser into a Langchain chain
chain = prompt | llm | output_parser

# Run the Langchain chain only if the user entered input text
if input_text:
    # Get the model response to the user's question
    response = chain.invoke({"question": input_text})

    # Display the model response on the Streamlit app
    st.write(response)
