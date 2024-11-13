import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Arrow

def draw_flowchart():
    fig, ax = plt.subplots(figsize=(6, 4))

    # Define as caixas com cores e textos
    boxes = [
        {"xy": (0.5, 0.8), "text": "Defina os parâmetros do\nmodelo.", "color": "#FFA07A"},
        {"xy": (0.5, 0.5), "text": "Treine o modelo", "color": "#DA70D6"},
        {"xy": (0.5, 0.2), "text": "Visualize as métricas", "color": "#98FB98"},
    ]

    # Adiciona as caixas
    for box in boxes:
        ax.add_patch(
            FancyBboxPatch(
                box["xy"], 0.4, 0.15, boxstyle="round,pad=0.05",
                linewidth=1, edgecolor="black", facecolor=box["color"]
            )
        )
        ax.text(
            box["xy"][0] + 0.2, box["xy"][1] + 0.075, box["text"], 
            ha="center", va="center", fontsize=10, color="black"
        )

    # Adiciona setas entre as caixas
    ax.add_patch(Arrow(0.7, 0.76, 0, -0.15, width=0.03, color="black"))
    ax.add_patch(Arrow(0.7, 0.46, 0, -0.15, width=0.03, color="black"))

    # Adiciona seta de retorno para o topo
    ax.add_patch(Arrow(0.85, 0.2, 0, 0.55, width=0.03, color="black"))
    ax.add_patch(Arrow(0.85, 0.75, -0.3, 0, width=0.03, color="black"))

    # Adiciona seta e texto para a barra lateral
    ax.add_patch(Arrow(0.1, 0.4, 0.2, 0, width=0.05, color="black"))
    ax.text(0.02, 0.4, "Utilize a barra lateral\npara navegar entre as páginas", 
            ha="center", va="center", fontsize=10)

    # Ajusta os limites e remove os eixos
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    return fig

# Exibe o diagrama no Streamlit
st.pyplot(draw_flowchart())
