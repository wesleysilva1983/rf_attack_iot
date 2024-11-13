import streamlit as st
import pandas as pd

# Configuração do seletor de páginas na barra lateral
st.sidebar.title("Navegação")

# Criando botões na barra lateral para cada página
if st.sidebar.button("Instruções"):
    st.session_state.page = "instructions"
if st.sidebar.button("Configuração de Hiperparâmetros"):
    st.session_state.page = "hyperparameters"
if st.sidebar.button("Resultados"):
    st.session_state.page = "results"

# Definindo a página padrão caso ainda não exista uma página selecionada na sessão
if "page" not in st.session_state:
    st.session_state.page = "instructions"  # Página padrão

# Carregar dados uma única vez para todas as páginas
if "X_train" not in st.session_state:
    st.session_state.X_train = pd.read_csv('dataset/X_train.csv')
    st.session_state.y_train = pd.read_csv('dataset/y_train.csv').values.ravel()
    st.session_state.X_test = pd.read_csv('dataset/X_test.csv')
    st.session_state.y_test = pd.read_csv('dataset/y_test.csv').values.ravel()

# Carregar a página correspondente com base no estado da sessão
if st.session_state.page == "instructions":
    import modules.instructions as instructions
    instructions.show()
elif st.session_state.page == "hyperparameters":
    import modules.hyperparameters as hyperparameters
    hyperparameters.show()
elif st.session_state.page == "results":
    import modules.results as results
    results.show()