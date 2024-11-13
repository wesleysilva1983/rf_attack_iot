import streamlit as st
import streamlit as st
import os

# Read files *.py
file_path = os.path.join("code", "description.py")
with open(file_path) as f:
    exec(f.read())

# Read files *.py
file_path = os.path.join("code", "model_ml.py")
with open(file_path) as f:
    exec(f.read())

