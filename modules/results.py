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

        # Exibir o boxplot das métricas de desempenho
        st.subheader("Boxplot das Métricas de Desempenho")
        fig_boxplot, ax1 = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=metrics_df, ax=ax1)
        ax1.set_title("Boxplot das Métricas de Desempenho (Treino x Teste)")
        st.pyplot(fig_boxplot)

        # Preparar os dados para o gráfico de barras horizontal individualizado
        train_counts = pd.Series(st.session_state['y_train']).value_counts().sort_index()
        test_counts = pd.Series(st.session_state['y_test']).value_counts().sort_index()

        # Renomear as classes com as etiquetas específicas
        class_counts_df = pd.DataFrame({
            "Treino": train_counts,
            "Teste": test_counts
        }).fillna(0)
        class_counts_df.index = ["0 (benign)", "1 (mirai)", "2 (gafgyt)"]

        # Exibir o gráfico de barras horizontal individualizado
        st.subheader("Quantidade de Dados por Classe (Treino x Teste)")
        fig_bar, ax2 = plt.subplots(figsize=(10, 6))
        class_counts_df.plot(kind='barh', ax=ax2, width=0.7)
        ax2.set_xlabel("Quantidade")
        ax2.set_ylabel("Classe")
        ax2.legend(title="Conjunto de Dados", loc="lower right")
        st.pyplot(fig_bar)

        # Exibir a matriz de confusão e o classification report para a última rodada
        st.subheader("Matriz de Confusão e Classification Report")

        # Matriz de Confusão
        y_pred_test = rf.predict(st.session_state.X_test)
        conf_matrix = confusion_matrix(st.session_state.y_test, y_pred_test)
        fig_confusion, ax3 = plt.subplots(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax3)
        ax3.set_xlabel("Predição")
        ax3.set_ylabel("Real")
        ax3.set_title("Matriz de Confusão")
        st.pyplot(fig_confusion)

        # Classification Report com escala logarítmica
        report = classification_report(st.session_state.y_test, y_pred_test, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_df = report_df.drop(["accuracy", "macro avg", "weighted avg"], errors="ignore")

        st.subheader("Classification Report por Classe (Escala Logarítmica)")
        fig_class_report, ax4 = plt.subplots(figsize=(10, 6))
        report_df[["precision", "recall", "f1-score"]].plot(kind="bar", ax=ax4)
        ax4.set_title("Classification Report (Escala Logarítmica)")
        ax4.set_ylabel("Score")
        ax4.set_xlabel("Classe")
        ax4.set_yscale("log")
        ax4.legend(loc="lower right")
        ax4.yaxis.tick_right()
        st.pyplot(fig_class_report)

    else:
        st.warning("Por favor, vá para a página de Configuração de Hiperparâmetros e treine o modelo antes de visualizar os resultados.")
