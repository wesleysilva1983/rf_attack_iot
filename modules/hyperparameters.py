import streamlit as st
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def show():
    st.title("Configuração de Hiperparâmetros")

    # Hiperparâmetros configuráveis
    n_estimators = st.slider("Número de Árvores (n_estimators)", min_value=1, max_value=30, step=5, value=10)
    criterion = st.selectbox("Critério de Divisão (criterion)", options=["gini", "entropy"], index=1)
    max_features = st.selectbox("Máximo de Features (max_features)", options=["auto", "sqrt", "log2"], index=1)
    max_depth = st.slider("Profundidade Máxima das Árvores (max_depth)", min_value=3, max_value=15, step=3, value=6)
    min_samples_split = st.slider("Mínimo de Amostras para Dividir (min_samples_split)", min_value=3, max_value=15, step=3, value=6)

    # Botão para iniciar o treinamento
    if st.button("Treinar Modelo"):
        metrics = {"Acurácia Treino": [], "F1-Score Treino": [], "Acurácia Teste": [], "F1-Score Teste": []}
        
        with st.spinner("Aguarde... Treinando o modelo 10 vezes com diferentes random_states"):
            for i in range(10):
                rf = RandomForestClassifier(
                    n_estimators=n_estimators,
                    random_state=np.random.randint(0, 10000),  # Random_state diferente em cada rodada
                    criterion=criterion,
                    max_features=max_features,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split
                )
                rf.fit(st.session_state.X_train, st.session_state.y_train)
                
                # Armazenar o modelo e as métricas da última execução
                if i == 9:  # Última rodada
                    st.session_state["trained_model"] = rf
                
                # Fazer previsões e calcular as métricas
                y_pred_train = rf.predict(st.session_state.X_train)
                y_pred_test = rf.predict(st.session_state.X_test)
                
                # Armazenar métricas
                metrics["Acurácia Treino"].append(accuracy_score(st.session_state.y_train, y_pred_train))
                metrics["F1-Score Treino"].append(f1_score(st.session_state.y_train, y_pred_train, average="weighted"))
                metrics["Acurácia Teste"].append(accuracy_score(st.session_state.y_test, y_pred_test))
                metrics["F1-Score Teste"].append(f1_score(st.session_state.y_test, y_pred_test, average="weighted"))

        # Salvar métricas no estado da sessão para uso na página de resultados
        st.session_state["metrics"] = metrics
        st.success("Modelo treinado com sucesso! Vá para a página de Resultados para visualizar os resultados.")
