import pandas as pd
from pathlib import Path
import joblib

# ML libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# visualização
import matplotlib.pyplot as plt
import seaborn as sns

# loading pre-processed data
print("--- Carregando os dados pré-processados ---")
atual_path = Path(__file__).parent
pasta_dados_proc = atual_path / 'data_processed'

try:
    X = pd.read_pickle(pasta_dados_proc / 'X_final.pkl')
    y = pd.read_pickle(pasta_dados_proc / 'y_final.pkl')
    print("DataFrames X e y carregados com sucesso!")
    
    # divisao treino e teste
    # X contém tanto texto quanto números, vamos precisar separar
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Dados divididos: {len(X_train)} para treino, {len(X_test)} para teste.")

    print("\nDados prontos para a etapa de modelagem de regressão!")

    # comparação de modelos de regressão
    modelos = {
        "Regressão Linear": LinearRegression(),
        "Random Forest": RandomForestRegressor(random_state=42, n_jobs=-1), # n_jobs=-1 usa todos os processadores
        "Gradient Boosting": GradientBoostingRegressor(random_state=42)
    }

    resultados = []
    print("\n--- INICIANDO COMPARAÇÃO DE MODELOS DE REGRESSÃO ---")
    for nome, modelo in modelos.items():
        print(f"\nTreinando o modelo: {nome}...")
        modelo.fit(X_train, y_train)
        previsoes = modelo.predict(X_test)
        
        # métricas de regressão
        mae = mean_absolute_error(y_test, previsoes)
        r2 = r2_score(y_test, previsoes)
        
        resultados.append({
            "Modelo": nome,
            "MAE (Erro Médio $)": mae,
            "R² Score": r2
        })
        print(f"Resultados para {nome} calculados.")

    #  tabela de resultados
    df_resultados = pd.DataFrame(resultados)
    print("\n--- TABELA DE COMPARAÇÃO DE PERFORMANCE ---")
    print(df_resultados)

    # comparação para R²
    # print("\n--- Gerando gráfico de comparação (R² Score) ---")
    # plt.figure(figsize=(10, 6))
    # sns.barplot(data=df_resultados, x='Modelo', y='R² Score')
    # plt.title('Comparação de Modelos - R² Score')
    # plt.ylabel('Pontuação R² (Quanto mais perto de 1, melhor)')
    # plt.ylim(0, 1) # Fixa o eixo Y entre 0 e 1
    # plt.tight_layout()
    # plt.savefig('./imgs/comparacao_modelos_regressao_r2.png')
    # print("Gráfico salvo em './imgs/comparacao_modelos_regressao_r2.png'")

    # comparação para MAE
    # print("\n--- Gerando gráfico de comparação (MAE) ---")
    # plt.figure(figsize=(10, 6))
    # sns.barplot(data=df_resultados, x='Modelo', y='MAE (Erro Médio $)')
    # plt.title('Comparação de Modelos - Erro Médio Absoluto (MAE)')
    # plt.ylabel('Erro Médio em Dólares (Quanto menor, melhor)')
    # plt.tight_layout()
    # plt.savefig('./imgs/comparacao_modelos_regressao_mae.png')
    # print("Gráfico salvo em './imgs/comparacao_modelos_regressao_mae.png'")

    # salvando melhor modelo - Random Forest Regressor
    print("\n--- Treinando e salvando o modelo campeão (Random Forest Regressor) ---")
    modelo_campeao = RandomForestRegressor(random_state=42, n_jobs=-1)
    # treinando com todos os dados disponíveis
    modelo_campeao.fit(X_train, y_train) 
    
    # salvando
    pasta_saida = atual_path / 'final_model'
    pasta_saida.mkdir(exist_ok=True)
    joblib.dump(modelo_campeao, pasta_saida / 'modelo.joblib')
    
    # salvando a lista de colunas utilizadas no modelo
    lista_de_colunas = list(X_train.columns)
    joblib.dump(lista_de_colunas, pasta_saida / 'colunas_modelo.joblib')
    
    print("Modelo campeão e lista de colunas salvos com sucesso!")


except FileNotFoundError:
    print(f"ERRO: Arquivos .pkl não encontrados na pasta '{pasta_dados_proc}'.")
    print("Certifique-se de executar o script 'correlation_view.py' primeiro.")
except Exception as e:
    print(f"Ocorreu um erro inesperado: {e}")