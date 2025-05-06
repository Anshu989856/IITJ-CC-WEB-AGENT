import os
import sys
import torch
import types
import streamlit as st
from dynamic import answer_question  # Import your backend function

# Set environment variable
os.environ["STREAMLIT_WATCHER_IGNORE"] = "torch,torch.*"

# Create a monkey patch for PyTorch's classes module BEFORE importing streamlit
# This prevents the file watcher from raising an error when accessing torch.classes.__path__
class MockPath:
    _path = []
    
# Apply the monkey patch to torch.classes
if hasattr(torch, 'classes'):
    if not hasattr(torch.classes, '__path__'):
        torch.classes.__path__ = MockPath()

# Initialize the app
st.set_page_config(page_title="IITJ CC Website Assistant", page_icon="💻", layout="centered")

# Set custom CSS for a colorful UI
st.markdown("""
    <style>
        body {
            background: linear-gradient(135deg, #ff7e5f, #feb47b);
            font-family: 'Arial', sans-serif;
        }
        .stTextArea textarea {
            font-size: 18px;
            padding: 15px;
            border-radius: 10px;
            background-color: #ffffff;
            border: 2px solid #ff7e5f;
            color: #333;
            font-weight: bold;
        }
        .stTextArea textarea:focus {
            outline: none;
            border-color: #feb47b;
        }
        .stButton button {
            background-color: #ff7e5f;
            color: white;
            font-size: 18px;
            border-radius: 12px;
            padding: 15px 30px;
            transition: background-color 0.3s, transform 0.3s;
            font-weight: bold;
        }
        .stButton button:hover {
            background-color: #feb47b;
            transform: scale(1.1);
        }
        .stSpinner svg {
            color: #ff7e5f;
        }
        .main-title {
            color: #ffffff;
            font-family: 'Helvetica', sans-serif;
            text-align: center;
            font-size: 45px;
            font-weight: bold;
            margin-top: 30px;
        }
        .sub-title {
            font-size: 22px;
            text-align: center;
            margin-bottom: 40px;
            color: #ffffff;
            font-family: 'Arial', sans-serif;
            font-weight: lighter;
        }
        .stMarkdown h3 {
            color: #ff7e5f;
            font-size: 24px;
            font-weight: bold;
        }
        .stTextArea textarea {
            font-size: 18px;
            padding: 15px;
        }
    </style>
""", unsafe_allow_html=True)

# Title and description with bright colors
st.markdown("<div class='main-title'>💻 IITJ CC Website Assistant</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Ask any question about the IITJ CC website!</div>", unsafe_allow_html=True)

# Create an input field for the query
query = st.text_area("Enter your query here:", height=100, placeholder="Type your question here...")

# Display a vibrant button with a hover effect
if st.button("Ask"):
    if query.strip() == "":
        st.warning("Please enter a query first.")
    else:
        with st.spinner('Thinking...'):
            answer = answer_question(query)
        st.success("Answer generated!")
        
        # Response area with vibrant styling
        st.markdown("### 📜 Response:")
        st.write(answer)
