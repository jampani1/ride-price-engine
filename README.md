# 🚗 Previsão de Preços de Corridas (Uber & Lyft)
### <p align="center">Projeto de Regressão</p>

## Visão Geral do Projeto
Este projeto constrói um pipeline completo de Machine Learning para prever o preço de corridas de serviços como Uber e Lyft, utilizando um dataset público de Boston. O objetivo foi realizar uma análise exploratória, aplicar técnicas de engenharia de features e comparar múltiplos modelos de **Regressão** para encontrar a solução mais precisa para estimar o valor de uma corrida.

## 🛠️ Tecnologias Utilizadas
- **Linguagem:** Python
- **Bibliotecas:** Pandas, Scikit-learn, Matplotlib, Seaborn, Joblib

---

## 🔬 Pipeline do Projeto

#### 1. Análise Exploratória de Dados (EDA)
A investigação começou com a análise das features mais intuitivas.
- **Distância vs. Preço:** Um gráfico de dispersão mostrou que, apesar da correlação positiva, a distância sozinha não explica a grande variação nos preços.
- **Tipo de Serviço vs. Preço:** Um boxplot revelou "degraus" de preços bem definidos entre os diferentes tipos de serviço (ex: `Shared` vs. `Lux Black XL`), confirmando esta como uma feature fundamental.

![Gráfico de Dispersão de Distância vs. Preço][distancia_preco]
![Boxplot de Tipo de Serviço vs. Preço][tiposervico_preco]

#### 2. Engenharia de Features
Para enriquecer o modelo, novas features foram criadas:
- **Features de Tempo:** A coluna `timestamp` foi convertida em variáveis mais úteis, como `hora`, `dia_da_semana` e `mês`. A análise subsequente revelou padrões claros, como picos de preço nos horários de pico (8h e 17h).
- **Variáveis Categóricas:** Features de texto como `cab_type` e `name` foram transformadas em formato numérico usando **One-Hot Encoding** (`pd.get_dummies()`).

#### 3. Comparação de Múltiplos Modelos de Regressão
Com os dados preparados, foi realizado um "campeonato" entre três modelos de regressão para identificar o mais performático.
- **Regressão Linear**
- **Random Forest Regressor**
- **Gradient Boosting Regressor**

As métricas de avaliação utilizadas foram:
- **MAE (Erro Médio Absoluto):** O erro médio da previsão em dólares. Quanto menor, melhor.
- **R² (Coeficiente de Determinação):** O poder explicativo do modelo. Quanto mais perto de 1 (100%), melhor.

---

## 📈 Resultados e Análise

### Performance dos Modelos
A análise comparativa mostrou a clara superioridade dos modelos baseados em árvores.

| Modelo | MAE (Erro Médio $) | R² Score |
| :--- | :--- | :--- |
| Regressão Linear | $1.77 | 0.927 |
| **Random Forest** | **$1.26** | 0.957 |
| Gradient Boosting | $1.28 | **0.959** |

Embora o Gradient Boosting tenha o maior R², o **Random Forest foi escolhido como o modelo final** por apresentar o **menor MAE**. Em um contexto de negócio, minimizar o erro médio em dólares é uma métrica mais direta e interpretável.

![Gráfico de Barras de MAE por Modelo][comparacao_mae]
![Gráfico de Barras de R² por Modelo][comparacao_r2]

---

## 🚀 Como Executar o Projeto

O projeto é dividido em dois scripts principais para um fluxo de trabalho profissional:

1.  **`treinamento_e_analise.py`:** Executa todo o pipeline de EDA, limpeza, treinamento, comparação e salva o modelo vencedor.
2.  **`predict.py`:** Carrega o modelo salvo e o utiliza para prever preços de novas corridas.

### Pré-requisitos
- Python 3.8+
- Bibliotecas listadas em `requirements.txt`

### Instalação e Execução
1. Clone o repositório: `git clone https://github.com/jampani1/ride-price-engine.git`
2. Instale as dependências: `pip install -r requirements.txt`
3. Execute o script de treinamento: `python treinamento_e_analise.py`
4. Execute o script de previsão para ver um exemplo: `python predict.py`

---

## 🔮 Melhorias Futuras
- **Engenharia de Features Geográficas:** Utilizar as colunas `source` e `destination` para criar features como distância do centro da cidade ou agrupamento por bairros.
- **Inclusão de Dados Climáticos:** Aplicar técnicas de "imputação" para preencher os dados climáticos faltantes e testar seu impacto no modelo.
- **Otimização de Hiperparâmetros:** Usar `GridSearchCV` para encontrar a combinação ótima de parâmetros para o Random Forest.

---

## 📄 Fonte dos Dados
- **Dataset:** [Uber and Lyft Dataset Boston, MA](https://www.kaggle.com/datasets/brllrb/uber-and-lyft-dataset-boston-ma)

---

## 👨‍💻 Conecte-se Comigo

Este projeto foi desenvolvido por mim, **Maurício J Souza**, como parte da minha jornada de aprendizado em ciência de dados e machine learning. Para considerações, perguntas ou oportunidades, sinta-se à vontade para me encontrar em:

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mauriciojampani/)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/jampani1)
[![Gmail](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:mmjampani13@gmail.com)

[distancia_preco]: imgs/distancia_preco.png
[tiposervico_preco]: imgs/tiposervico_preco.png
[comparacao_mae]: imgs/comparacao_modelos_regressao_mae.png
[comparacao_r2]: imgs/comparacao_modelos_regressao_r2.png
