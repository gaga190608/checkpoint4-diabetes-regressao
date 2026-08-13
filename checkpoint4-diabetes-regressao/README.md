# Checkpoint 4 — Regressão Linear e Polinomial

**Disciplina:** Data Science & Statistical Computing — FIAP 2026
**Tema:** Progressão da diabetes em função de indicadores clínicos

## Objetivo

Este projeto investiga em que medida indicadores clínicos (IMC, pressão arterial, colesterol e glicemia)
explicam a progressão da diabetes em pacientes, um ano após um exame inicial (baseline). O projeto cobre
todo o fluxo de um problema de regressão linear: formulação do problema, tratamento e exploração dos dados,
modelagem (referência, simples, múltipla e polinomial), diagnóstico do modelo e uma aplicação interativa em
Streamlit para gerar novas previsões.

## Origem dos dados

- **Base:** Diabetes Dataset (Efron, Hastie, Johnstone & Tibshirani, 2004).
- **Fonte:** distribuída publicamente via `scikit-learn` (`sklearn.datasets.load_diabetes`), com URL de
  origem documentada em https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html.
- **Tamanho:** 442 pacientes, 10 variáveis explicativas + 1 variável resposta.
- O arquivo `dados/diabetes.csv` já contém uma cópia exportada da base (não é necessário baixar nada
  manualmente).

## Estrutura dos arquivos

```
projeto/
├── app.py                 # Aplicação Streamlit
├── notebook.ipynb          # Notebook com a análise completa (EDA, modelagem, diagnóstico)
├── requirements.txt        # Dependências do projeto
├── README.md
├── dados/
│   └── diabetes.csv        # Base de dados (já tratada/exportada)
└── modelo/
    ├── modelo.pkl           # Modelo final (Pipeline: ColumnTransformer + LinearRegression) treinado
    └── metadados.pkl        # Intervalos observados das variáveis e métricas de teste, usados no app
```

## Instalação das dependências

```bash
pip install -r requirements.txt
```

## Execução do notebook

Abra `notebook.ipynb` no Jupyter (ou `jupyter nbconvert --to notebook --execute notebook.ipynb` para
reexecutar do zero). O notebook, ao final, salva o modelo treinado em `modelo/modelo.pkl` e os metadados em
`modelo/metadados.pkl` — arquivos que a aplicação Streamlit consome diretamente.

## Execução da aplicação

```bash
streamlit run app.py
```

A aplicação carrega `dados/diabetes.csv` e `modelo/modelo.pkl` (cacheados com `@st.cache_data` e
`@st.cache_resource`), mostra estatísticas descritivas, gráficos exploratórios, as métricas do modelo final
no conjunto de teste (MAE, RMSE, R²) e um formulário para gerar novas previsões. A entrada do usuário passa
pelo mesmo pipeline de pré-processamento usado no treinamento (`ColumnTransformer` + `LinearRegression`), e
a aplicação avisa quando algum valor informado está fora do intervalo observado nos dados de treino
(extrapolação).

## Modelo final escolhido

Foram comparados 4 modelos no mesmo conjunto de teste (30% dos dados, `random_state=42`):

| Modelo | MAE | RMSE | R² |
|---|---|---|---|
| Referência (média) | 64,26 | 73,71 | -0,006 |
| Regressão simples (bmi) | 50,59 | 62,33 | 0,280 |
| **Regressão múltipla (todas as variáveis)** | **41,92** | **53,12** | **0,477** |
| Regressão polinomial (bmi grau 2 + demais) | 42,05 | 53,22 | 0,475 |

A regressão polinomial não trouxe ganho relevante sobre a múltipla puramente linear — por isso, o **modelo
de regressão múltipla** foi escolhido como final, por ser mais simples e igualmente eficaz.

## Principais limitações conhecidas

- Amostra relativamente pequena (442 pacientes) e de uma única coorte.
- Ausência de variáveis de estilo de vida, histórico familiar ou tempo de diagnóstico prévio, que
  provavelmente melhorariam o modelo.
- A variável `sex` é fornecida pela fonte original apenas como códigos numéricos (1/2), sem legenda pública
  de qual valor corresponde a qual sexo — por isso foi tratada como categoria genérica ("Grupo_1"/"Grupo_2").
- O modelo é um estudo observacional: as associações encontradas **não implicam causalidade**.
- Previsões para valores de entrada fora do intervalo observado na base de treino são extrapolações e devem
  ser interpretadas com cautela (a aplicação Streamlit sinaliza esses casos automaticamente).

## Uso de inteligência artificial

Partes deste projeto (estruturação do notebook, redação de texto e código da aplicação) foram desenvolvidas
com apoio de IA, com base no enunciado do Checkpoint 4. Recomenda-se que quem for apresentar o projeto
revise e compreenda integralmente o código e as decisões de modelagem antes da apresentação.
