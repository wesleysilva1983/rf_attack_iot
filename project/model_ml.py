import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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
y_pred = rf.predict(X_test)

# Avaliar o modelo
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

# Exibir os resultados
st.write("**Acurácia:**", accuracy)
st.write("**Relatório de Classificação:**")
st.dataframe(pd.DataFrame(report).transpose())

# Plotar a matriz de confusão
st.write("**Matriz de Confusão:**")
conf_matrix = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
ax.set_xlabel("Predicted Labels")
ax.set_ylabel("True Labels")
ax.set_title("Confusion Matrix")
st.pyplot(fig)
