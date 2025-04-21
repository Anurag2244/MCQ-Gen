import traceback
import os
import pandas as pd
from dotenv import load_dotenv
from src.mcqgenerator.utils import read_file, get_table_data
from src.mcqgenerator.logger import logging

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain

load_dotenv()
key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(openai_api_key = key, model_name='gpt-4o-mini', temperature=0.5)

TEMPLATE = """
Text:{text}
You are a MCQ maker. Given above text, it is your job to create a quiz of {number} multiple choice questions on {subject} in {level} level.
Make sure the questions are not repeated and check all questions to confirm the text as well.
Make sure to format your response like RESPONSE_JSON below and use it as a guide.
Ensure to make {number} MCQs
### RESPONSE_JSON
{response_json}
"""

quiz_generation_prompt = PromptTemplate(
    input_variables=['text', 'number', 'subject', 'tone', 'response_json'],
    template=TEMPLATE
)

quiz_chain = LLMChain(llm=llm, prompt=quiz_generation_prompt, output_key='quiz', verbose=True)

TEMPLATE_2="""
You are an english grammarian and writer. Given a multiple choice questions on {subject} students.
You need to evaluate the complexity of the questions and give a complete analysis of the quiz. Only use at max 50 words for complexity analysis. 
if the quiz is not at per with the cognitive and analytical abilities of the students,
update the quiz questions which needs to be changed and change the tone such that it perfectly fits the student abilities
Quiz_MCQs:
{quiz}

Check as an expert English Writer for the above quiz:
"""

quiz_evaluation_prompt=PromptTemplate(input_variables=["subject", "quiz"], template=TEMPLATE_2)

review_chain = LLMChain(llm=llm, prompt=quiz_evaluation_prompt, output_key="review", verbose=True)

generate_evaluate_chain=SequentialChain(chains=[quiz_chain, review_chain], input_variables=["text", "number", "subject", "level", "response_json"],
                                        output_variables=["quiz", "review"], verbose=True)
