import streamlit as st
import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
import io

load_dotenv()
groq_api_key = os.environ.get("GROQ_API_KEY")

llm = ChatGroq(temperature=0, model_name="llama3-70b-8192", api_key=groq_api_key)

summary_prompt_single = PromptTemplate.from_template("""
Here is the data for the student:
The given structured data is complex, but the structure can be broken down as follows:
1. Column 1: user_id – A unique identifier for each student.
2. Columns 2 to 9: subject_scores – A dictionary of subject names as keys and the corresponding scores as values. Each student may have attempted multiple subjects or chapters, which are represented here. The columns alternate between subject names and their respective scores.
3. Column 10: productivity_yes_no – Indicates whether the student was considered productive (Yes/No).
4. Column 11: productivity_rate – A numerical value representing the student's productivity level, typically ranging from 1 to 10.
5. Column 12: emotional_factors – Provides information about any emotional factors that may have impacted the student's performance, such as "EMOTIONAL FACTORS" or "BACKLOGS". This column helps highlight specific concerns affecting the student.(academic_panic_buttons = ("MISSED CLASSES", "BACKLOGS", "LACK OF MOTIVATION", "NOT UNDERSTANDING", "BAD MARKS"), non_academic_panic_buttons = ("EMOTIONAL FACTORS", "PROCRASTINATE", "LOST INTEREST", "LACK OF FOCUS", "GOALS NOT ACHIEVED", "LACK OF DISCIPLINE"))

{context}

Based on this data, generate a descriptive summary of the student's strengths, opportunities, and challenges.
Also provide some specific suggestions on how the student can improve. Avoid generic statements.
""")

summary_prompt_multiple = PromptTemplate.from_template("""
Here is the data for the students:
The given structured data is complex, but the structure can be broken down as follows:
1. Column 1: user_id – A unique identifier for each student.
2. Columns 2 to 9: subject_scores – A dictionary of subject names as keys and the corresponding scores as values. Each student may have attempted multiple subjects or chapters, which are represented here. The columns alternate between subject names and their respective scores.
3. Column 10: productivity_yes_no – Indicates whether the student was considered productive (Yes/No).
4. Column 11: productivity_rate – A numerical value representing the student's productivity level, typically ranging from 1 to 10.
5. Column 12: emotional_factors – Provides information about any emotional factors that may have impacted the student's performance, such as "EMOTIONAL FACTORS" or "BACKLOGS". This column helps highlight specific concerns affecting the student.(academic_panic_buttons = ("MISSED CLASSES", "BACKLOGS", "LACK OF MOTIVATION", "NOT UNDERSTANDING", "BAD MARKS"), non_academic_panic_buttons = ("EMOTIONAL FACTORS", "PROCRASTINATE", "LOST INTEREST", "LACK OF FOCUS", "GOALS NOT ACHIEVED", "LACK OF DISCIPLINE"))

{context}

Generate a detailed summary of the strengths, opportunities, and challenges for these students.
Provide specific insights for each student, and compare their strengths and areas for improvement.
Suggest ways they can learn from each other and address their challenges collaboratively where applicable.
""")

@st.cache_data
def load_data():
    # Replace this with the actual file path where your CSV file is stored
    file_path = 'https://raw.githubusercontent.com/forittik/Student_collaboration_task1/refs/heads/main/merged_3dataset.csv'  # Change 'your_file_path_here.csv' to the actual path

    # Load the data from the CSV file
    df = pd.read_csv(file_path, header=None)
    
    # Assign column names dynamically based on the number of columns
    num_columns = len(df.columns)
    column_names = ['user_id']
    for i in range(1, num_columns - 3):  # -3 for the last three columns
        if i % 2 == 1:
            column_names.append(f'subject_{(i+1)//2}')
        else:
            column_names.append(f'score_{i//2}')
    column_names.extend(['productivity_yes_no', 'productivity_rate', 'emotional_factors'])
    
    df.columns = column_names
    
    return df
def get_student_data(name, df):
    student_data = df[df['user_id'] == name]
    if student_data.empty:
        return None
    return student_data

def generate_single_student_summary(student_data):
    context = student_data.to_string(index=False)
    summary_chain = summary_prompt_single | llm | StrOutputParser()
    summary = summary_chain.invoke({"context": context})
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
student_names = df['user_id'].tolist()
selected_names = st.multiselect("Select student(s) to analyze:", student_names)

if st.button("Analyze student data"):
    if selected_names:
        summary = process_students(selected_names, df)
        st.write(summary)
    else:
        st.warning("Please select at least one student.")
