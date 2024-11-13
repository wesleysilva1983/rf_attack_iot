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

        # Exibir o boxplot das métricas de desempenho e gráfico de barras horizontal
        st.subheader("Boxplot das Métricas e Quantitativo de Dados do Dataset")

        # Preparar os dados para o gráfico de barras individualizado
        train_counts = pd.Series(st.session_state['y_train']).value_counts().sort_index()
        test_counts = pd.Series(st.session_state['y_test']).value_counts().sort_index()

        # Criar um DataFrame para facilitar a plotagem
        class_counts_df = pd.DataFrame({
            "Treino": train_counts,
            "Teste": test_counts
        }).fillna(0)

        # Configurar o layout dos gráficos lado a lado
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [2, 1]})

        # Boxplot das métricas
        sns.boxplot(data=metrics_df, ax=ax1)
        ax1.set_title("Boxplot das Métricas de Desempenho (Treino x Teste)")

        # Gráfico de barras horizontal com barras individualizadas para cada classe
        class_counts_df.plot(kind='barh', ax=ax2, width=0.7)
        ax2.set_title("Quantidade de Dados por Classe (Treino x Teste)")
        ax2.set_xlabel("Quantidade")
        ax2.set_ylabel("Classe")
        ax2.legend(title="Conjunto de Dados", loc="lower right")

        # Exibir o gráfico na interface
        st.pyplot(fig)

        # Exibir a matriz de confusão e o classification report para a última rodada
        st.subheader("Matriz de Confusão e Classification Report")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Matriz de Confusão
        y_pred_test = rf.predict(st.session_state.X_test)
        conf_matrix = confusion_matrix(st.session_state.y_test, y_pred_test)
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax1)
        ax1.set_xlabel("Predição")
        ax1.set_ylabel("Real")
        ax1.set_title("Matriz de Confusão")

        # Classification Report com escala logarítmica
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