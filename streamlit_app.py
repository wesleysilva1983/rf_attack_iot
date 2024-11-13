import streamlit as st
import os
import pandas as pd

# Configuração de navegação
st.sidebar.title("Navegação")
page = st.sidebar.selectbox("Selecione a Página", ["Instruções", "Configuração de Hiperparâmetros", "Resultados"])

# Carregar dados uma única vez para todas as páginas
if "X_train" not in st.session_state:
    st.session_state.X_train = pd.read_csv('dataset/X_train.csv')
    st.session_state.y_train = pd.read_csv('dataset/y_train.csv').values.ravel()
    st.session_state.X_test = pd.read_csv('dataset/X_test.csv')
    st.session_state.y_test = pd.read_csv('dataset/y_test.csv').values.ravel()

# Carregar as páginas com base na seleção
if page == "Instruções":
    import modules.instructions as instructions
    instructions.show()
elif page == "Configuração de Hiperparâmetros":
    import modules.hyperparameters as hyperparameters
    hyperparameters.show()
elif page == "Resultados":
    import modules.results as results
    results.show()