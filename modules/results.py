import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

def show():
    st.title("Resultados do Modelo")

    # Verificar se o modelo foi treinado e as métricas estão salvas
    if "trained_model" in st.session_state and "metrics" in st.session_state:
        rf = st.session_state["trained_model"]
        metrics_df = pd.DataFrame(st.session_state["metrics"])

        # Exibir os hiperparâmetros do modelo
        st.subheader("Hiperparâmetros Utilizados no Treinamento")
        st.write(f"**Número de Árvores (n_estimators):** {rf.n_estimators}")
        st.write(f"**Critério (criterion):** {rf.criterion}")
        st.write(f"**Máximo de Features (max_features):** {rf.max_features}")
        st.write(f"**Profundidade Máxima (max_depth):** {rf.max_depth}")
        st.write(f"**Mínimo de Amostras para Dividir (min_samples_split):** {rf.min_samples_split}")

        # Gráfico 1: Quantitativo de Dados por Classe
        st.subheader("Quantidade de Dados por Classe (Treino x Teste)")
        train_counts = pd.Series(st.session_state['y_train']).value_counts().sort_index()
        test_counts = pd.Series(st.session_state['y_test']).value_counts().sort_index()

        class_counts_df = pd.DataFrame({
            "Treino": train_counts,
            "Teste": test_counts
        }).fillna(0)

        fig, ax = plt.subplots(figsize=(8, 6))
        bars = class_counts_df.plot(kind='barh', ax=ax, width=0.7)
        ax.set_xlabel("Quantidade")
        ax.set_ylabel("Classe")
        ax.set_title("Quantidade de Dados por Classe (Treino x Teste)")
        ax.legend(title="Conjunto de Dados", loc="lower right")

        # Adiciona as labels no centro das barras
        for container in ax.containers:
            ax.bar_label(container, label_type='center')
        
        st.pyplot(fig)

        # Gráfico 2: Boxplot das Métricas de Desempenho
        st.subheader("Boxplot das Métricas de Desempenho (Treino x Teste)")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(data=metrics_df, ax=ax)
        ax.set_title("Boxplot das Métricas de Desempenho")
        st.pyplot(fig)

        # Gráfico 3: Classification Report
        st.subheader("Classification Report (Escala Logarítmica)")
        y_pred_test = rf.predict(st.session_state.X_test)
        report = classification_report(st.session_state.y_test, y_pred_test, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_df = report_df.drop(["accuracy", "macro avg", "weighted avg"], errors="ignore")

        fig, ax = plt.subplots(figsize=(8, 6))
        report_df[["precision", "recall", "f1-score"]].plot(kind="bar", ax=ax)
        ax.set_title("Classification Report")
        ax.set_ylabel("Score")
        ax.set_xlabel("Classe")
        ax.set_yscale("log")
        ax.legend(loc="lower right")
        st.pyplot(fig)

        # Gráfico 4: Matriz de Confusão
        st.subheader("Matriz de Confusão")
        conf_matrix = confusion_matrix(st.session_state.y_test, y_pred_test)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set_xlabel("Predição")
        ax.set_ylabel("Real")
        ax.set_title("Matriz de Confusão")
        st.pyplot(fig)

    else:
        st.warning("Por favor, vá para a página de Configuração de Hiperparâmetros e treine o modelo antes de visualizar os resultados.")
