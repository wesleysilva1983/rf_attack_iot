import streamlit as st

def show():
    st.markdown("""
    ### Pós-graduação em Ciência da Computação - UEL
    **Projeto 2COP507:** Detecção de Ataques no Tráfego de Dispositivos IoT.
                                   
    **Discente:** Wesley Candido da Silva.
            
    **Dataset:** N-BaIoT Dataset to Detect IoT Botnet Attacks
          
    **Link github:** https://github.com/wesleysilva1983/rf_attack_iot
    """)

    st.markdown("""
    ### Classificador Random Forest
                
    **Funcionalidades do aplicativo:**
                
    1. Configuração dos hiperparâmetros para o modelo RF.
    2. Treinamento do modelo.
    3. Métricas do modelo.
                
    # Carregando a imagem de um arquivo local
    st.image("images/fluxo.JPG", caption="Fluxo para utilizar o App", use_column_width=True)
                
    """)
    

