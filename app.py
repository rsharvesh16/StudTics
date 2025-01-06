import os
import boto3
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from scipy import stats

load_dotenv()

# Initialize Bedrock client
bedrock = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

def data_ingestion(uploaded_files):
    all_data = []
    for uploaded_file in uploaded_files:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        if file_extension == '.csv':
            df = pd.read_csv(uploaded_file)
        elif file_extension in ['.xlsx', '.xls']:
            df = pd.read_excel(uploaded_file)
        else:
            st.error(f"Unsupported file format: {file_extension}")
            continue
        
        text = f"File: {uploaded_file.name}\n"
        text += f"Columns: {', '.join(df.columns)}\n"
        text += df.to_string(index=False)
        all_data.append(text)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.create_documents(all_data)
    return docs

def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    vectorstore_faiss.save_local("faiss_index")

def get_mistral_llm():
    llm = Bedrock(model_id="mistral.mistral-7b-instruct-v0:2", client=bedrock,
                  model_kwargs={'max_tokens': 8000, 'temperature': 0.2})
    return llm

prompt_template = """
Human: You are an AI assistant for college professors, tasked with analyzing student data and providing insights. Use the following context, which contains relevant information from the student data files, to answer the question. Follow these guidelines strictly:

1. Provide comprehensive answers based solely on the information present in the given context.
2. If the context doesn't contain enough information to answer the question fully, state what information is missing and answer with the parts you can address accurately.
3. If you're unsure about any part of the answer, explicitly state your uncertainty and provide the information you are confident about.
4. If the question cannot be answered at all based on the given context, clearly state that you don't have enough information to provide an accurate answer.
5. Do not make assumptions or include information that isn't explicitly stated in the context.
6. Always mention the specific columns, data points, or file names that you're using to form your answer.
7. Provide detailed, thorough responses that fully address all aspects of the question.
8. When analyzing a student's performance, consider their attendance, marks in various subjects, and any other relevant data to provide a comprehensive overview.
9. Suggest areas of improvement for the student based on their performance data.
10. Highlight the student's strengths based on their academic record.
11. If available, comment on the student's performance trend over time.
12. Provide actionable recommendations for the professor to help the student improve.

Context: {context}

Question: {question}
Assistant: Based on the student data provided, I'll provide a comprehensive analysis:

"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

def get_response_llm(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(
            search_type="similarity", search_kwargs={"k": 5}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa({"query": query})
    return answer['result']

def load_student_data(uploaded_files):
    all_data = pd.DataFrame()
    for uploaded_file in uploaded_files:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        if file_extension == '.csv':
            df = pd.read_csv(uploaded_file)
        elif file_extension in ['.xlsx', '.xls']:
            df = pd.read_excel(uploaded_file)
        else:
            st.error(f"Unsupported file format: {file_extension}")
            continue
        all_data = pd.concat([all_data, df], ignore_index=True)
    return all_data

def get_student_data(df, identifier, id_type):
    if id_type == "Registration Number":
        return df[df["Registration Number"] == identifier]
    elif id_type == "Name":
        return df[df["Name"] == identifier]
    else:
        return pd.DataFrame()

def plot_attendance(student_data):
    subjects = [col for col in student_data.columns if col.endswith("Attendance")]
    attendance_data = student_data[subjects].iloc[0]
    
    df = pd.DataFrame({
        "Subject": subjects,
        "Attendance": attendance_data.values
    })
    
    fig = px.bar(df, x="Subject", y="Attendance", 
                 labels={"Subject": "Subject", "Attendance": "Attendance %"},
                 title="Attendance Percentage by Subject")
    fig.update_layout(xaxis_tickangle=-45)
    return fig

def plot_marks(student_data):
    subjects = [col for col in student_data.columns if col.endswith("Marks")]
    marks_data = student_data[subjects].iloc[0]
    
    df = pd.DataFrame({
        "Subject": subjects,
        "Marks": marks_data.values
    })
    
    fig = px.bar(df, x="Subject", y="Marks", 
                 labels={"Subject": "Subject", "Marks": "Marks"},
                 title="Marks by Subject")
    fig.update_layout(xaxis_tickangle=-45)
    return fig

def plot_radar_chart(student_data):
    subjects = [col for col in student_data.columns if col.endswith("Marks")]
    marks_data = student_data[subjects].iloc[0]
    
    fig = go.Figure(data=go.Scatterpolar(
        r=marks_data,
        theta=subjects,
        fill='toself'
    ))
    fig.update_layout(title="Subject Performance Overview")
    return fig

def calculate_class_rank(student_df, student_data):
    total_marks = student_df[[col for col in student_df.columns if col.endswith("Marks")]].sum(axis=1)
    sorted_df = total_marks.sort_values(ascending=False)
    student_total = student_data[[col for col in student_data.columns if col.endswith("Marks")]].sum().iloc[0]
    
    # Find the rank
    rank = (sorted_df > student_total).sum() + 1
    
    return rank, len(student_df)
def calculate_percentile(student_df, student_data):
    total_marks = student_df[[col for col in student_df.columns if col.endswith("Marks")]].sum(axis=1)
    student_total = student_data[[col for col in student_data.columns if col.endswith("Marks")]].sum().iloc[0]
    percentile = stats.percentileofscore(total_marks, student_total)
    return percentile

def plot_performance_trend(student_data):
    subjects = [col for col in student_data.columns if col.endswith("Marks")]
    marks_data = student_data[subjects].iloc[0]
    
    fig = go.Figure()
    for subject in subjects:
        fig.add_trace(go.Scatter(x=student_data['Year'], y=student_data[subject], mode='lines+markers', name=subject))
    
    fig.update_layout(title="Performance Trend Over Years", xaxis_title="Year", yaxis_title="Marks")
    return fig

def main():
    st.set_page_config(page_title="Advanced Student Analytics Dashboard", layout="wide")

    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
            body {
                font-family: 'Roboto', sans-serif;
                background-color: #f0f2f6;
            }
            .container {
                background-color: #ffffff;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                max-width: 1200px;
                margin: 0 auto;
            }
            .title {
                text-align: center;
                color: #1e3a8a;
                font-weight: 700;
                font-size: 2.5rem;
                margin-bottom: 20px;
            }
            .upload-container {
                padding: 20px;
                border: 2px dashed #3b82f6;
                border-radius: 10px;
                text-align: center;
                background-color: #e0f2fe;
                margin-bottom: 30px;
            }
            .sidebar-title {
                font-size: 1.2rem;
                font-weight: bold;
                color: #1e40af;
                margin-bottom: 15px;
            }
            .footer {
                text-align: center;
                margin-top: 40px;
                font-size: 0.9rem;
                color: #64748b;
            }
            .stat-card {
                background-color: #f0f9ff;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
                margin-bottom: 20px;
            }
            .stat-card h3 {
                color: #0369a1;
                font-size: 1.2rem;
                margin-bottom: 10px;
            }
            .stat-card p {
                font-size: 2rem;
                font-weight: bold;
                color: #0c4a6e;
            }
        </style>
        <div class="container">
            <h1 class="title">Advanced Student Analytics Dashboard</h1>
            <p>Upload student files (CSV, Excel) and analyze individual student performance with AI-powered insights.</p>
        </div>
    """, unsafe_allow_html=True)

    st.write("<br>", unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader("Upload student data files", type=["csv", "xlsx", "xls"], accept_multiple_files=True)
    
    if uploaded_files:
        with st.spinner("Processing files..."):
            docs = data_ingestion(uploaded_files)
            get_vector_store(docs)
            st.success("Files processed and knowledge base updated successfully!")
        
        student_df = load_student_data(uploaded_files)
        
        # Student search
        search_type = st.radio("Search by:", ("Registration Number", "Name"))
        search_input = st.text_input(f"Enter student {search_type}:")
        
        if search_input:
            student_data = get_student_data(student_df, search_input, search_type)
            
            if not student_data.empty:
                st.write(f"## Student Information: {student_data['Name'].iloc[0]}")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown('<div class="stat-card">'
                                f'<h3>Registration Number</h3>'
                                f'<p>{student_data["Registration Number"].iloc[0]}</p>'
                                '</div>', unsafe_allow_html=True)
                with col2:
                    st.markdown('<div class="stat-card">'
                                f'<h3>Class</h3>'
                                f'<p>{student_data["Class"].iloc[0]}</p>'
                                '</div>', unsafe_allow_html=True)
                with col3:
                    st.markdown('<div class="stat-card">'
                                f'<h3>Year</h3>'
                                f'<p>{student_data["Year"].iloc[0]}</p>'
                                '</div>', unsafe_allow_html=True)
                with col4:
                    st.markdown('<div class="stat-card">'
                                f'<h3>Mobile Number</h3>'
                                f'<p>{student_data["Mobile Number"].iloc[0]}</p>'
                                '</div>', unsafe_allow_html=True)
                
                # Attendance chart
                st.plotly_chart(plot_attendance(student_data), use_container_width=True)
                
                # Marks chart
                st.plotly_chart(plot_marks(student_data), use_container_width=True)
                
                # Radar chart
                st.plotly_chart(plot_radar_chart(student_data), use_container_width=True)
                
                # Performance trend
                st.plotly_chart(plot_performance_trend(student_data), use_container_width=True)
                
                # LLM Analysis
                st.write("## AI-Powered Analysis")
                llm = get_mistral_llm()
                faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
                
                analysis_question = f"""Provide a comprehensive analysis of the student {search_input}, including:
                1. Their strengths and areas for improvement based on their academic performance.
                2. Notable patterns in their attendance and marks across different subjects.
                3. Their overall performance trend over the years.
                4. A comparison of their performance with the class average.
                5. Actionable recommendations for the professor to help the student improve.
                6. Suggestions for personalized learning strategies based on the student's performance data.
                """
                
                analysis = get_response_llm(llm, faiss_index, analysis_question)
                
                st.write(analysis)
                
                # Comparison with class average
                st.write("## Comparison with Class Average")
                subjects = [col for col in student_df.columns if col.endswith("Marks")]
                student_marks = student_data[subjects].iloc[0]
                class_average = student_df[subjects].mean()
                
                comparison_df = pd.DataFrame({
                    "Subject": subjects,
                    "Student Marks": student_marks.values,
                    "Class Average": class_average.values
                })
                
                comparison_df_melted = pd.melt(comparison_df, id_vars=['Subject'], var_name='Category', value_name='Marks')
                
                fig = px.bar(comparison_df_melted, x="Subject", y="Marks", color="Category",
                             barmode="group", title="Student Performance vs Class Average")
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
                
                # Improvement suggestions
                st.write("## Personalized Improvement Suggestions")
                improvement_question = f"""Based on the performance data of student {search_input}, provide:
                1. Three specific areas where the student needs improvement.
                2. Three concrete strategies for each area that the professor can implement to help the student.
                3. Two personalized study techniques that could benefit this student based on their learning patterns.
                4. One recommendation for extracurricular activities that could complement their academic strengths.
                """
                
                improvement_suggestions = get_response_llm(llm, faiss_index, improvement_question)
                st.write(improvement_suggestions)
                
                # Parent-Teacher Meeting Notes Generator
                st.write("## Parent-Teacher Meeting Notes Generator")
                if st.button("Generate Parent-Teacher Meeting Notes"):
                    meeting_notes_question = f"""Create a concise set of talking points for a parent-teacher meeting regarding student {search_input}. Include:
                    1. A brief overview of the student's academic performance.
                    2. Highlights of their strengths and areas needing improvement.
                    3. Specific recommendations for parental support at home.
                    4. Any concerns about attendance or behavior (if applicable).
                    5. Positive notes about the student's progress or potential.
                    """
                    
                    meeting_notes = get_response_llm(llm, faiss_index, meeting_notes_question)
                    st.write(meeting_notes)
                
                # Export Report
                st.write("## Export Student Report")
                if st.button("Generate Exportable Report"):
                    report_question = f"""Create a comprehensive report for student {search_input} that includes:
                    1. Executive summary of the student's overall performance.
                    2. Detailed breakdown of performance in each subject.
                    3. Attendance analysis and its impact on academic performance.
                    4. Comparison with class averages.
                    5. Identified strengths and areas for improvement.
                    6. Recommended action plans for the student, parents, and teachers.
                    7. Projected academic trajectory based on current performance.
                    """
                    
                    report = get_response_llm(llm, faiss_index, report_question)
                    st.download_button(
                        label="Download Student Report",
                        data=report,
                        file_name=f"{student_data['Name'].iloc[0]}_report.txt",
                        mime="text/plain"
                    )
            
            else:
                st.warning(f"No student found with the given {search_type}. Please check and try again.")
    
    else:
        st.info("Please upload student data files to begin analysis.")
    
    st.markdown('<div class="footer">Powered by Mistral LLM and Streamlit</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()