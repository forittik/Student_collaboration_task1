import streamlit as st
import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
load_dotenv()
groq_api_key = os.environ.get("GROQ_API_KEY")

llm = ChatGroq(temperature=0, model_name="llama3-70b-8192", api_key=groq_api_key)

summary_prompt_single = PromptTemplate.from_template("""
Here is the data for the student:

{context}

Based on this data, generate a descriptive summary of the student's strengths, opportunities, and challenges.
Also provide some specific suggestions on how the student can improve. Avoid generic statements.
""")

summary_prompt_multiple = PromptTemplate.from_template("""
Here is the data for the students:

{context}

Generate a detailed summary of the strengths, opportunities, and challenges for these students.
Provide specific insights for each student, and compare their strengths and areas for improvement.
Suggest ways they can learn from each other and address their challenges collaboratively where applicable.
""")

@st.cache_data
def load_data():
    return pd.read_csv("https://raw.githubusercontent.com/forittik/Student_collaboration_task1/refs/heads/main/student_scores.csv")

def get_student_data(name, df):
    student_data = df[df.iloc[:, 0] == name]
    if student_data.empty:
        return None
    return student_data

def generate_single_student_summary(student_data):
    context = student_data.to_string(index=False)
    summary_chain = summary_prompt_single | llm | StrOutputParser() #The LLM’s output is often complex (such as JSON-like structures or dictionaries), but the StrOutputParser() simplifies it to just the string (text) portion that you want.
    summary = summary_chain.invoke({"context": context}) #This calls (or "invokes") the summary_chain with a dictionary {"context": context} as the input.
    return summary

def generate_multiple_students_summary(student_data):
    context = student_data.to_string(index=False)
    summary_chain = summary_prompt_multiple | llm | StrOutputParser()
    summary = summary_chain.invoke({"context": context})
    return summary

def process_students(names, df):
    if isinstance(names, str):
        student_data = get_student_data(names, df)
        if student_data is None:
            return f"No data found for student: {names}"
        return generate_single_student_summary(student_data)
    elif isinstance(names, list):
        combined_data = pd.concat([get_student_data(name, df) for name in names if get_student_data(name, df) is not None])
        if combined_data.empty:
            return "No data found for the given students."
        return generate_multiple_students_summary(combined_data)

st.title("B2B Dashboard")
df = load_data()
student_names = df.iloc[:, 0].unique().tolist()
selected_names = st.multiselect("Select student(s) to generate analyze:", student_names)

if st.button("Analyze student data"):
    if selected_names:
        summary = process_students(selected_names, df)
        st.write(summary)
    else:
        st.warning("Please select at least one student.")
