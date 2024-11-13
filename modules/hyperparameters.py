import streamlit as st
from sklearn.ensemble import RandomForestClassifier

def show():
    st.title("Configuração de Hiperparâmetros")

    # Hiperparâmetros configuráveis
    n_estimators = st.slider("Número de Árvores (n_estimators)", min_value=1, max_value=30, step=5, value=10)
    random_state = st.slider("Estado Aleatório (random_state)", min_value=12, max_value=96, step=12, value=36)
    criterion = st.selectbox("Critério de Divisão (criterion)", options=["gini", "entropy"], index=1)
    max_features = st.selectbox("Máximo de Features (max_features)", options=["auto", "sqrt", "log2"], index=1)
    max_depth = st.slider("Profundidade Máxima das Árvores (max_depth)", min_value=3, max_value=15, step=3, value=6)
    min_samples_split = st.slider("Mínimo de Amostras para Dividir (min_samples_split)", min_value=3, max_value=15, step=3, value=6)

    # Botão para iniciar o treinamento
    if st.button("Treinar Modelo"):
        with st.spinner("Aguarde... Treinando o modelo"):
            rf = RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=random_state,
                criterion=criterion,
                max_features=max_features,
                max_depth=max_depth,
                min_samples_split=min_samples_split
            )
            rf.fit(st.session_state.X_train, st.session_state.y_train)
        
        st.session_state["trained_model"] = rf
        st.success("Modelo treinado com sucesso! Vá para a página de Resultados para visualizar os resultados.")
