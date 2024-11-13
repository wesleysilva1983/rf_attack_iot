import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Carregar os dados
X_train = pd.read_csv('dataset/X_train.csv')
y_train = pd.read_csv('dataset/y_train.csv').values.ravel()  # Para transformar em uma array 1D
X_test = pd.read_csv('dataset/X_test.csv')
y_test = pd.read_csv('dataset/y_test.csv').values.ravel()

# Configurar o modelo de Random Forest
st.title("Random Forest Classifier")

n_estimators = st.slider("Número de Árvores (n_estimators)", min_value=10, max_value=200, value=100)
max_depth = st.slider("Profundidade Máxima das Árvores (max_depth)", min_value=1, max_value=20, value=5)

# Treinar o modelo
rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
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