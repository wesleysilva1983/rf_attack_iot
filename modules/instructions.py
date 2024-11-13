import streamlit as st

def show():
    st.title("Instruções")
    st.write("""
    Bem-vindo ao Classificador Random Forest! Este aplicativo permite que você:
    1. Configure hiperparâmetros para o modelo Random Forest.
    2. Treine o modelo com os dados de exemplo.
    3. Veja o desempenho do modelo e visualize a matriz de confusão.
    
    Utilize a barra lateral para navegar entre as páginas.
    """)
