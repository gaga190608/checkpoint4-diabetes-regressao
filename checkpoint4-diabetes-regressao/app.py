"""
Checkpoint 4 - Regressão Linear e Polinomial
Aplicação Streamlit - Progressão da Diabetes em Função de Indicadores Clínicos
FIAP - 2026
"""

import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Progressão da Diabetes", layout="wide")

# --------------------------------------------------------------------------------------
# Carregamento de dados, modelo e metadados (cacheado para não recarregar a cada interação)
# --------------------------------------------------------------------------------------
@st.cache_data
def carregar_dados():
    return pd.read_csv("dados/diabetes.csv")


@st.cache_resource
def carregar_modelo():
    with open("modelo/modelo.pkl", "rb") as f:
        modelo = pickle.load(f)
    with open("modelo/metadados.pkl", "rb") as f:
        metadados = pickle.load(f)
    return modelo, metadados


df = carregar_dados()
modelo, metadados = carregar_modelo()

# --------------------------------------------------------------------------------------
# Cabeçalho
# --------------------------------------------------------------------------------------
st.title("Progressão da Diabetes — Regressão Linear")
st.markdown(
    """
    **Problema:** em que medida indicadores clínicos (IMC, pressão arterial, marcadores de colesterol e
    glicemia) explicam a progressão da diabetes um ano após o exame inicial (baseline)?

    **Fonte dos dados:** Diabetes Dataset (Efron, Hastie, Johnstone & Tibshirani, 2004),
    distribuído publicamente via `scikit-learn` (`sklearn.datasets.load_diabetes`).

    **Variável resposta:** `progresso_doenca` — medida quantitativa contínua da progressão da doença.
    **Variáveis usadas na previsão:** idade, sexo (grupo), IMC, pressão arterial média e seis marcadores
    de sangue (colesterol total, LDL, HDL, razão colesterol/HDL, triglicerídeos e glicemia).
    """
)

st.divider()

# --------------------------------------------------------------------------------------
# Amostra e estatísticas descritivas
# --------------------------------------------------------------------------------------
st.header("1. Amostra da base e estatísticas descritivas")

col1, col2 = st.columns([2, 1])
with col1:
    st.dataframe(df.head(10))
with col2:
    st.metric("Nº de pacientes", df.shape[0])
    st.metric("Progressão média", f"{df['progresso_doenca'].mean():.1f}")
    st.metric("Progressão (desvio padrão)", f"{df['progresso_doenca'].std():.1f}")

st.dataframe(df.describe().round(2))

# --------------------------------------------------------------------------------------
# Gráficos exploratórios
# --------------------------------------------------------------------------------------
st.header("2. Gráficos exploratórios")

g1, g2 = st.columns(2)
with g1:
    fig, ax = plt.subplots()
    ax.hist(df["progresso_doenca"], bins=20, color="steelblue")
    ax.set_title("Distribuição da progressão da doença")
    ax.set_xlabel("Progressão da doença (escore clínico)")
    ax.set_ylabel("Frequência")
    st.pyplot(fig)

with g2:
    fig, ax = plt.subplots()
    ax.scatter(df["bmi"], df["progresso_doenca"], alpha=0.6, color="darkorange")
    ax.set_title("Progressão da doença vs. IMC")
    ax.set_xlabel("IMC (kg/m²)")
    ax.set_ylabel("Progressão da doença")
    st.pyplot(fig)

st.divider()

# --------------------------------------------------------------------------------------
# Métricas do modelo final
# --------------------------------------------------------------------------------------
st.header("3. Desempenho do modelo final (regressão múltipla)")

m1, m2, m3 = st.columns(3)
m1.metric("MAE", f"{metadados['mae_teste']:.2f}")
m2.metric("RMSE", f"{metadados['rmse_teste']:.2f}")
m3.metric("R²", f"{metadados['r2_teste']:.3f}")

st.caption(
    "Métricas calculadas no conjunto de teste (30% da base, não usado no treinamento do modelo)."
)

# Gráficos de diagnóstico (real vs previsto e resíduos) recalculados a partir da base completa,
# usando o mesmo pipeline de pré-processamento do treinamento.
X_full = df.drop(columns=["progresso_doenca"])
y_full = df["progresso_doenca"]
pred_full = modelo.predict(X_full)
residuos_full = y_full - pred_full

d1, d2 = st.columns(2)
with d1:
    fig, ax = plt.subplots()
    ax.scatter(y_full, pred_full, alpha=0.5, color="seagreen")
    lims = [min(y_full.min(), pred_full.min()), max(y_full.max(), pred_full.max())]
    ax.plot(lims, lims, "r--", label="Previsão perfeita")
    ax.set_title("Valores reais vs. previstos")
    ax.set_xlabel("Progressão real")
    ax.set_ylabel("Progressão prevista")
    ax.legend()
    st.pyplot(fig)

with d2:
    fig, ax = plt.subplots()
    ax.scatter(pred_full, residuos_full, alpha=0.5, color="indianred")
    ax.axhline(0, color="black", linestyle="--")
    ax.set_title("Resíduos vs. valores ajustados")
    ax.set_xlabel("Valor previsto")
    ax.set_ylabel("Resíduo")
    st.pyplot(fig)

st.divider()

# --------------------------------------------------------------------------------------
# Formulário de previsão
# --------------------------------------------------------------------------------------
st.header("4. Faça uma previsão")
st.markdown("Informe os valores clínicos do paciente para estimar a progressão da doença em 1 ano.")

intervalos = metadados["intervalos"]
categorias_sex = metadados["categorias_sex"]

with st.form("form_previsao"):
    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.number_input(
            "Idade (anos)",
            min_value=0.0, max_value=120.0,
            value=float(df["age"].median()),
        )
        sex = st.selectbox("Grupo (sex)", categorias_sex)
        bmi = st.slider(
            "IMC (kg/m²)",
            min_value=float(intervalos["bmi"][0] * 0.7),
            max_value=float(intervalos["bmi"][1] * 1.3),
            value=float(df["bmi"].median()),
        )

    with c2:
        bp = st.slider(
            "Pressão arterial média (mmHg)",
            min_value=float(intervalos["bp"][0] * 0.7),
            max_value=float(intervalos["bp"][1] * 1.3),
            value=float(df["bp"].median()),
        )
        s1 = st.number_input("Colesterol total (s1, mg/dL)", value=float(df["s1"].median()))
        s2 = st.number_input("LDL (s2, mg/dL)", value=float(df["s2"].median()))

    with c3:
        s3 = st.number_input("HDL (s3, mg/dL)", value=float(df["s3"].median()))
        s4 = st.number_input("Colesterol total / HDL (s4)", value=float(df["s4"].median()))
        s5 = st.number_input("Log dos triglicerídeos (s5)", value=float(df["s5"].median()))
        s6 = st.number_input("Glicemia (s6, mg/dL)", value=float(df["s6"].median()))

    enviar = st.form_submit_button("Prever progressão da doença")

if enviar:
    entrada = pd.DataFrame(
        [{
            "age": age, "sex": sex, "bmi": bmi, "bp": bp,
            "s1": s1, "s2": s2, "s3": s3, "s4": s4, "s5": s5, "s6": s6,
        }]
    )

    # Aviso de extrapolação: mesma checagem de intervalo usada para toda variável numérica,
    # comparando com o intervalo observado nos dados de TREINO (registrado em metadados).
    fora_intervalo = []
    for col in metadados["colunas_num"]:
        minimo, maximo = intervalos[col]
        valor = float(entrada[col].iloc[0])
        if valor < minimo or valor > maximo:
            fora_intervalo.append(f"{col} (observado entre {minimo:.1f} e {maximo:.1f}, informado {valor:.1f})")

    if fora_intervalo:
        st.warning(
            "⚠️ Atenção: os seguintes valores estão **fora do intervalo observado** nos dados de treino, "
            "então a previsão abaixo é uma extrapolação e deve ser interpretada com cautela:\n\n- "
            + "\n- ".join(fora_intervalo)
        )

    # A entrada passa pelo MESMO pipeline (ColumnTransformer + LinearRegression) usado no treino.
    previsao = modelo.predict(entrada)[0]
    st.success(f"**Progressão estimada da doença: {previsao:.1f} (escore clínico)**")
    st.caption(
        "Este resultado é uma estimativa estatística baseada em dados históricos — não substitui "
        "avaliação médica."
    )

st.divider()
st.caption(
    "Checkpoint 4 - Data Science & Statistical Computing - Prof. Jones Egydio - FIAP 2026"
)
