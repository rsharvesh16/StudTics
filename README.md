# StudTics

The *StudTics* is an AI-powered Streamlit application designed to help college professors analyze student performance data. The dashboard enables the uploading of student data files and provides detailed insights into student performance, attendance, and trends using interactive visualizations and advanced AI analysis.

---

## Features

1. **File Upload and Processing**
   - Supports `.csv` and `.xlsx` file formats.
   - Automatically processes and ingests student data into a knowledge base for further analysis.

2. **AI-Powered Insights**
   - Uses Amazon Bedrock models for embeddings and text generation.
   - Provides detailed and actionable insights into student performance and areas for improvement.

3. **Visualizations**
   - Attendance and marks are visualized using bar charts.
   - A radar chart offers an overview of subject-wise performance.
   - Line plots display performance trends over the years.
   - Comparison of individual performance with the class average.

4. **Search and Analyze**
   - Search students by their registration number or name.
   - Get detailed student information, including class, year, and performance stats.

5. **Class Performance Metrics**
   - Calculates class rank and percentile for students.
   - Highlights performance trends and areas of improvement.

6. **Customized Styling**
   - Clean and user-friendly interface with responsive design.
   - Dynamic themes for presenting data effectively.

---

## Tech Stack

- **Frontend:** Streamlit
- **Backend:** Python
- **AI Models:** Amazon Bedrock (Mistral, Titan Embeddings)
- **Visualizations:** Plotly
- **Data Storage:** FAISS (for vector search)

---

## How to Use

1. **Upload Files**
   - Upload one or more student data files in `.csv` or `.xlsx` format.

2. **Search for Students**
   - Use the search functionality to look up a student by their registration number or name.

3. **View Insights**
   - Explore attendance and marks visualizations.
   - Get personalized AI-driven analysis of student performance.

4. **Compare with Class**
   - View individual performance compared to the class average.
   - Get class rank and percentile information.

5. **Actionable Recommendations**
   - Access suggestions for improving student performance and learning strategies.

---

## Acknowledgments

- **Amazon Bedrock**: For providing powerful embeddings and text-generation models.
- **LangChain**: For simplifying AI pipeline integrations.
- **Streamlit**: For an interactive and easy-to-use interface.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---
