import os
from dotenv import load_dotenv
load_dotenv()  # make sure to call it

from langchain.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

# Set environment variables
##os.environ["langchain_api_key"] = os.getenv("langchain_api_key")

##os.environ["langchain_project"] = os.getenv("langchain_project")

# Initialize Ollama LLM
llm = Ollama(model="gemma:2b")

# Create prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant, please respond to the question asked."),
    ("user", "Question:{question}")  # placeholder matches the key in invoke
])

# Output parser
output_parser = StrOutputParser()

# Build chain
chain = prompt | llm | output_parser

# Streamlit UI
st.title("LangChain Demo with Gemma 2b")
input_text = st.text_input("What question is in your mind?")

if input_text:
    response = chain.invoke({"question": input_text})
    st.write(response)



## cd DataIngestion
## streamlit run App.py    ---must do these two steps
##streamlit run Ap.py


