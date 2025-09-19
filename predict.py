import pandas as pd
from pathlib import Path
import joblib

# data preparation
def preparar_novos_dados(df, colunas_do_modelo):
    print("Preparando novos dados para o formato do modelo...")
    
    # features de tempo
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.weekday
    df['month'] = df['datetime'].dt.month
    
    # one-hot encoding para variaveis categóricas
    df_dummies = pd.get_dummies(df[['cab_type', 'name']])
    df_final = pd.concat([df, df_dummies], axis=1)
    
    # alinhamento de colunas com o modelo
    # .reindex() para forçar o df a ter as mesmas colunas, na mesma ordem, do modelo treinado.
    # colunas que não existirem nos novos dados serão criadas e preenchidas com 0.
    df_alinhado = df_final.reindex(columns=colunas_do_modelo, fill_value=0)
    
    return df_alinhado

if __name__ == "__main__":
    try:
        # carregar o modelo treinado e as colunas esperadas
        print("Carregando o modelo treinado e as colunas esperadas")
        atual_path = Path(__file__).parent
        pasta_modelo = atual_path / 'final_model'
        modelo = joblib.load(pasta_modelo / 'modelo.joblib')
        colunas_modelo = joblib.load(pasta_modelo / 'colunas_modelo.joblib')
        
        # carrega o dataset COMPLETO para criar a amostragem
        print("Carregando dataset completo para criar amostragem...")
        df_total = pd.read_csv(atual_path / 'data_source' / 'rideshare_kaggle.csv')
        
        # separa dados com e sem preço e criar uma amostra de demonstração
        df_com_preco = df_total[df_total['price'].notna()]
        df_sem_preco = df_total[df_total['price'].isna()]
        
        amostra_verificacao = df_com_preco.sample(10, random_state=42)
        amostra_previsao = df_sem_preco.sample(10, random_state=42)
        
        df_demonstracao = pd.concat([amostra_verificacao, amostra_previsao], ignore_index=True)
        print("Amostra de demonstração com 20 corridas criada.")
        
        # prepara os dados da amostra de demonstração
        X_pronto = preparar_novos_dados(df_demonstracao.copy(), colunas_modelo)
        
        # faz a previsão de preços para a amostra
        precos_previstos = modelo.predict(X_pronto)

        # cria e exibe o resultado COMPARATIVO
        print("\n--- RELATÓRIO DE DEMONSTRAÇÃO E PREVISÃO ---")
        resultado = pd.DataFrame({
            'Tipo': ['Verificação'] * 10 + ['Previsão'] * 10,
            'Serviço': df_demonstracao['name'],
            'Distância': df_demonstracao['distance'].round(2),
            'Hora': df_demonstracao['hour'],
            'Preço Real ($)': df_demonstracao['price'],
            'Preço Estimado ($)': precos_previstos.round(2)
        })
        
        # a coluna de 'Diferença' para os casos de verificação
        resultado['Diferença ($)'] = (resultado['Preço Estimado ($)'] - resultado['Preço Real ($)']).round(2)
        
        print(resultado)
        
        # salva este relatório em um novo arquivo CSV
        nome_arquivo_saida = 'relatorio.csv'
        resultado.to_csv(atual_path / nome_arquivo_saida, index=False)
        print(f"\nRelatório de demonstração salvo como '{nome_arquivo_saida}'")

    except Exception as e:
        print(f"Ocorreu um erro: {e}")