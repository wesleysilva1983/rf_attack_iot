import streamlit as st
import pandas as pd

# Configuração dos links de navegação
st.title("Bem-vindo ao Classificador Random Forest!")
st.markdown("### Navegação:")
st.markdown("[1. Instruções](?page=instructions)")
st.markdown("[2. Configuração de Hiperparâmetros](?page=hyperparameters)")
st.markdown("[3. Resultados](?page=results)")

# Obter o parâmetro da página da URL
query_params = st.experimental_get_query_params()
page = query_params.get("page", ["instructions"])[0]  # Valor padrão é "instructions" se o parâmetro não existir

# Carregar dados uma única vez para todas as páginas
if "X_train" not in st.session_state:
    st.session_state.X_train = pd.read_csv('dataset/X_train.csv')
    st.session_state.y_train = pd.read_csv('dataset/y_train.csv').values.ravel()
    st.session_state.X_test = pd.read_csv('dataset/X_test.csv')
    st.session_state.y_test = pd.read_csv('dataset/y_test.csv').values.ravel()

# Carregar a página correspondente com base no parâmetro
if page == "instructions":
    import modules.instructions as instructions
    instructions.show()
elif page == "hyperparameters":
    import modules.hyperparameters as hyperparameters
    hyperparameters.show()
elif page == "results":
    import modules.results as results
    results.show()