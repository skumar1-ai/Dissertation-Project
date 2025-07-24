This project is a Streamlit-based interactive dashboard for detecting anomalies in the Olist e-commerce dataset. It includes multiple components such as data cleaning, exploratory data analysis (EDA), payment anomaly detection, delivery anomaly detection, and review anomaly detection.

A built-in LLM-powered chatbot (using OpenAI's GPT) assists users with natural language queries on the dataset.

🚀 Features
✅ Data Cleaning & Preprocessing

📈 Interactive EDA Visualizations (Plotly)

⚠️ Payment & Delivery Anomaly Detection

Isolation Forest, LOF, KMeans

⭐ Review Anomaly Detection

Sentiment mismatches, LDA topic modeling

🤖 LLM Chatbot (PandasAI + OpenAI)

🔎 Custom Filtering, Highlighting & Insights

🛠️ Tech Stack
Python

Streamlit

Pandas, NumPy

Plotly, Matplotlib

scikit-learn

PandasAI

OpenAI API (GPT)

📦 Installation
Clone the repository:

bash
Copy
Edit
git clone https://github.com/yourusername/olist-anomaly-dashboard.git
cd olist-anomaly-dashboard
(Optional but recommended) Create a virtual environment:

bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install the required libraries:

bash
Copy
Edit
pip install -r requirements.txt
Set up your OpenAI API key:

Get your API key from OpenAI

Create a .env file in the root folder:

ini
Copy
Edit
OPENAI_API_KEY=your_openai_key_here
🧠 Using the Dashboard
Run the Streamlit app:

bash
Copy
Edit
streamlit run app.py
Then open the link shown in your terminal (usually http://localhost:8501) in a web browser.


