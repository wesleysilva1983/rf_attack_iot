import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

def show():
    st.title("Resultados do Modelo")

    # Verificar se o modelo foi treinado
    if "trained_model" in st.session_state:
        rf = st.session_state["trained_model"]

        # Exibir os hiperparâmetros do modelo
        st.subheader("Hiperparâmetros Utilizados no Treinamento")
        st.write(f"**Número de Árvores (n_estimators):** {rf.n_estimators}")
        st.write(f"**Critério (criterion):** {rf.criterion}")
        st.write(f"**Máximo de Features (max_features):** {rf.max_features}")
        st.write(f"**Profundidade Máxima (max_depth):** {rf.max_depth}")
        st.write(f"**Estado Aleatório (random_state):** {rf.random_state}")
        st.write(f"**Mínimo de Amostras para Dividir (min_samples_split):** {rf.min_samples_split}")

        # Armazenar as métricas de cada rodada
        metrics = {"Acurácia Treino": [], "F1-Score Treino": [], "Acurácia Teste": [], "F1-Score Teste": []}

        # Treinar o modelo 3 vezes e coletar as métricas
        for i in range(3):
            rf.fit(st.session_state.X_train, st.session_state.y_train)

            # Fazer previsões
            y_pred_train = rf.predict(st.session_state.X_train)
            y_pred_test = rf.predict(st.session_state.X_test)

            # Calcular métricas para treino e teste e armazenar
            metrics["Acurácia Treino"].append(accuracy_score(st.session_state.y_train, y_pred_train))
            metrics["F1-Score Treino"].append(f1_score(st.session_state.y_train, y_pred_train, average="weighted"))
            metrics["Acurácia Teste"].append(accuracy_score(st.session_state.y_test, y_pred_test))
            metrics["F1-Score Teste"].append(f1_score(st.session_state.y_test, y_pred_test, average="weighted"))

        # Converter as métricas em DataFrame para visualização
        metrics_df = pd.DataFrame(metrics)
        
        # Exibir as métricas de cada rodada como boxplots
        st.subheader("Boxplot das Métricas de Desempenho")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=metrics_df, ax=ax)
        ax.set_title("Boxplot das Métricas de Desempenho (Treino x Teste)")
        st.pyplot(fig)

        # Configurar figura e eixos para a matriz de confusão e o classification report
        st.subheader("Matriz de Confusão e Classification Report")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Matriz de Confusão (usando a última rodada de previsão para consistência)
        conf_matrix = confusion_matrix(st.session_state.y_test, y_pred_test)
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax1)
        ax1.set_xlabel("Predição")
        ax1.set_ylabel("Real")
        ax1.set_title("Matriz de Confusão")

        # Classification Report com escala logarítmica (última rodada)
        report = classification_report(st.session_state.y_test, y_pred_test, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_df = report_df.drop(["accuracy", "macro avg", "weighted avg"], errors="ignore")
        
        # Plotar o classification report como gráfico de barras com escala logarítmica
        report_df[["precision", "recall", "f1-score"]].plot(kind="bar", ax=ax2)
        ax2.set_title("Classification Report (Escala Logarítmica)")
        ax2.set_ylabel("Score")
        ax2.set_xlabel("Classe")
        ax2.set_yscale("log")
        ax2.legend(loc="lower right")
        ax2.yaxis.tick_right()

        # Exibir a figura com a matriz de confusão e o gráfico do classification report lado a lado
        st.pyplot(fig)

    else:
        st.warning("Por favor, vá para a página de Configuração de Hiperparâmetros e treine o modelo antes de visualizar os resultados.")