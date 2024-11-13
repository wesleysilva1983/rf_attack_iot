import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def show():
    st.title("Resultados do Modelo")

    # Verificar se o modelo foi treinado
    if "trained_model" in st.session_state:
        rf = st.session_state["trained_model"]

        # Fazer previsões
        y_pred_train = rf.predict(st.session_state.X_train)
        y_pred_test = rf.predict(st.session_state.X_test)

        # Calcular métricas para treino e teste
        accuracy_train = accuracy_score(st.session_state.y_train, y_pred_train)
        f1_score_train = f1_score(st.session_state.y_train, y_pred_train, average="weighted")
        accuracy_test = accuracy_score(st.session_state.y_test, y_pred_test)
        f1_score_test = f1_score(st.session_state.y_test, y_pred_test, average="weighted")

        # Exibir as métricas solicitadas
        st.write(f"**Acurácia (Treino):** {accuracy_train:.4f}")
        st.write(f"**F1-Score (Treino):** {f1_score_train:.4f}")
        st.write(f"**Acurácia (Teste):** {accuracy_test:.4f}")
        st.write(f"**F1-Score (Teste):** {f1_score_test:.4f}")

        # Plotar a matriz de confusão
        st.write("**Matriz de Confusão:**")
        conf_matrix = confusion_matrix(st.session_state.y_test, y_pred_test)

        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set_xlabel("Predicted Labels")
        ax.set_ylabel("True Labels")
        st.pyplot(fig)
    else:
        st.warning("Por favor, vá para a página de Configuração de Hiperparâmetros e treine o modelo antes de visualizar os resultados.")
