import os
import json
import traceback
import pandas as pd
from dotenv import load_dotenv
from src.mcqgenerator.utils import read_file, get_table_data
import streamlit as st
from langchain.callbacks import get_openai_callback
from src.mcqgenerator.MCQGenerator import generate_evaluate_chain
from src.mcqgenerator.logger import logging

with open('C:\Users\anura\OneDrive\Desktop\Gen AI project\MCQ Gen\response.json', 'r') as file:
    RESPONSE_JSON = json.load(file)

# Creating the title for the app
st.title("MCQ creator application with langchain")

# Create a form using st.form
with st.form("user_inputs"):
    #File upload
    uploaded_file=st.file_uploader("Upload a pdf or txt file")

    # Input fields
    mcq_count=st.number_input("No. of MCQs", min_value=3, max_value=50)

    # Subject
    subject=st.text_input("Insert Subject", max_chars=20)

    # Quiz tone
    level = st.text_input("Complexity level of questions", max_chars=20, placeholder="Simple")

    # Add button
    button = st.form_submit_button("Create MCQs")

    # Check if the buttons is clicked and all fields have input
    if button and uploaded_file is not None and mcq_count and subject and level:
        with st.spinner("loading..."):
            try:
                text = read_file(uploaded_file)

                with get_openai_callback() as cb:
                    response = generate_evaluate_chain(
                        {
                            "text": text,
                            "number": mcq_count,
                            "subject": subject,
                            "level": level,
                            "response_json": json.dumps(RESPONSE_JSON)
                        }
                    )
            except Exception as e:
                traceback.print_execution(type(e), e, e.__traceback__)
                st.error("Error")

            else:
                print(f"Total Tokens:{cb.total_tokens}")
                print(f"Prompt Tokens:{cb.prompt_tokens}")
                print(f"Completion Tokens:{cb.completion_tokens}")
                print(f"Total Cost: {cb.total_cost}")
                if isinstance(response, dict):
                    quiz=response.get("quiz", None)
                    if quiz is not None:
                        table_data = get_table_data(quiz)
                        if table_data is not None:
                            df = pd.DataFrame(table_data)
                            df.index = df.index+1
                            st.table(df)
                            st.text_area(label='Review', value=response["review"])
                        else:
                            st.error("Error in the table data")
                
                else:
                    st.write(response)
                  