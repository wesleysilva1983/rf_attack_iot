import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

# Carregar os dados
X_train = pd.read_csv('dataset/X_train.csv')
y_train = pd.read_csv('dataset/y_train.csv').values.ravel()  # Para transformar em uma array 1D
X_test = pd.read_csv('dataset/X_test.csv')
y_test = pd.read_csv('dataset/y_test.csv').values.ravel()

# Configurar o modelo de Random Forest
st.title("Random Forest Classifier")

# Hiperparâmetros configuráveis com os ajustes específicos
n_estimators = st.slider("Número de Árvores (n_estimators)", min_value=1, max_value=30, step=5, value=10)
random_state = st.slider("Estado Aleatório (random_state)", min_value=12, max_value=96, step=12, value=36)
criterion = st.selectbox("Critério de Divisão (criterion)", options=["gini", "entropy"], index=1)
max_features = st.selectbox("Máximo de Features (max_features)", options=["auto", "sqrt", "log2"], index=1)
max_depth = st.slider("Profundidade Máxima das Árvores (max_depth)", min_value=3, max_value=15, step=3, value=6)
min_samples_split = st.slider("Mínimo de Amostras para Dividir (min_samples_split)", min_value=3, max_value=15, step=3, value=6)

# Treinar o modelo
rf = RandomForestClassifier(
    n_estimators=n_estimators,
    random_state=random_state,
    criterion=criterion,
    max_features=max_features,
    max_depth=max_depth,
    min_samples_split=min_samples_split
)
rf.fit(X_train, y_train)

# Fazer previsões
y_pred_train = rf.predict(X_train)
y_pred_test = rf.predict(X_test)

# Calcular métricas para treino e teste
accuracy_train = accuracy_score(y_train, y_pred_train)
f1_score_train = f1_score(y_train, y_pred_train, average="weighted")
accuracy_test = accuracy_score(y_test, y_pred_test)
f1_score_test = f1_score(y_test, y_pred_test, average="weighted")

# Exibir as métricas solicitadas
st.write(f"**Acurácia (Treino):** {accuracy_train:.4f}")
st.write(f"**F1-Score (Treino):** {f1_score_train:.4f}")
st.write(f"**Acurácia (Teste):** {accuracy_test:.4f}")
st.write(f"**F1-Score (Teste):** {f1_score_test:.4f}")

# Plotar a matriz de confusão com tamanho reduzido
st.write("**Matriz de Confusão:**")
conf_matrix = confusion_matrix(y_test, y_pred_test)

fig, ax = plt.subplots(figsize=(4, 3))  # Ajuste o tamanho conforme necessário
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
ax.set_xlabel("Predicted Labels")
ax.set_ylabel("True Labels")
st.pyplot(fig)
