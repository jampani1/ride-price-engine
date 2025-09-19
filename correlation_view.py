# Script: analise_limpeza_inicial.py
# adicionando feature de pegar e tratar dados de hora e dia, transformar texto em numeros categoricos (tipo de servico: uberX (1), uberXL (2), etc)
# adicionando feature para one-hot encoding (cab_type e name), variaveis categoricas para variaveis numericas
# adicionando portabilidade (salvando df para modelagem em .csv para um novo script py)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

print("--- Iniciando a análise e limpeza inicial ---")
atual_path = Path(__file__).parent
csv_path = atual_path / 'data_source' / 'rideshare_kaggle.csv'

try:
    df = pd.read_csv(csv_path)
    print("Dataset carregado com sucesso!")

    # correção e limpeza da coluna 'price'
    # tipo original já é float, mas pode ter valores nulos ou inválidos
    print("\nVerificando o tipo de dado da coluna 'price'...")
    print(f"Tipo original: {df['price'].dtype}")

    # conversão forçada para um tipo numérico.
    # errors='coerce' é um truque útil: se algum valor não puder ser convertido, ele se tornará Nulo (NaN).
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    
    print(f"Tipo após conversão: {df['price'].dtype}")
    
    # AGORA que 'price' é numérico, podemos remover as linhas onde ele é Nulo.
    linhas_antes = len(df)
    df.dropna(subset=['price'], inplace=True)
    linhas_depois = len(df)
    
    print(f"\n{linhas_antes - linhas_depois} linhas com preço nulo foram removidas.")
    print(f"O dataset agora tem {linhas_depois} linhas.")
    
    # tudo isso para garantir que .describe() funcionará para a coluna 'price'!
    print("\nEstatísticas básicas da coluna 'price' após limpeza:")
    print(df['price'].describe())
    
    # relação entre distância e preço
    # print("\nGerando gráfico de Dispersão (Distância vs. Preço)...")
    # plt.figure(figsize=(12, 7))
    # sns.scatterplot(data=df, x='distance', y='price', alpha=0.5, s=15) # alpha e s para melhor visualização
    # plt.title('Relação entre Distância da Corrida e Preço', fontsize=16)
    # plt.xlabel('Distância (em milhas)', fontsize=12)
    # plt.ylabel('Preço (em $)', fontsize=12)
    # plt.grid(True)
    # plt.savefig('./imgs/distancia_preco.png')
    # print("Gráfico './imgs/distancia_preco.png' salvo com sucesso!")
    
    # relação entre tipo de serviço e preço 
    # print("\nGerando Boxplot (Tipo de Serviço vs. Preço)...")
    # plt.figure(figsize=(14, 8))
    # # Ordenamos os tipos de serviço pela mediana do preço para um gráfico mais legível
    # ordem = df.groupby('name')['price'].median().sort_values().index
    # sns.boxplot(data=df, x='name', y='price', order=ordem)
    # plt.title('Distribuição de Preços por Tipo de Serviço', fontsize=16)
    # plt.xlabel('Tipo de Serviço', fontsize=12)
    # plt.ylabel('Preço (em $)', fontsize=12)
    # plt.xticks(rotation=45, ha='right') # Rotaciona os nomes para não sobrepor
    # plt.tight_layout() # Ajusta o layout
    # plt.savefig('./imgs/tiposervico_preco.png')
    # print("Gráfico './imgs/tiposervico_preco.png' salvo com sucesso!")

    # data e hora
    print("\nCriando novas features a partir do timestamp...")
    
    # coluna 'timestamp' em um formato de data/hora que o Pandas entende.
    # unidade 's' é importante para dizer que o número representa segundos.
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    
    # extração das informações que queremos em novas colunas
    df['hour'] = df['datetime'].dt.hour
    df['day'] = df['datetime'].dt.day
    df['day_of_week'] = df['datetime'].dt.weekday  # 0=Segunda, 1=Terça, ..., 6=Domingo
    df['month'] = df['datetime'].dt.month
    
    print("Novas colunas de data/hora criadas com sucesso!")
    print("Visualização das novas colunas:")
    print(df[['datetime', 'hour', 'day', 'day_of_week', 'month']].head())
    print("\n")

    # análise de Features de Tempo vs. Preço
    #print("\nAnalisando a influência das novas features de tempo no preço...")

    # preço médio por hora do dia
    # plt.figure(figsize=(12, 7))
    # # usamos groupby() para agrupar todas as corridas por hora e calcular a média do preço para cada hora
    # preco_por_hora = df.groupby('hour')['price'].mean()
    # sns.lineplot(x=preco_por_hora.index, y=preco_por_hora.values, marker='o')
    # plt.title('Preço Médio da Corrida por Hora do Dia', fontsize=16)
    # plt.xlabel('Hora do Dia (0-23)', fontsize=12)
    # plt.ylabel('Preço Médio (em $)', fontsize=12)
    # plt.xticks(range(0, 24)) # Garante que todas as horas apareçam no eixo X
    # plt.grid(True)
    # plt.savefig('./imgs/hora_preco.png')
    # print("Gráfico './imgs/hora_preco.png' salvo com sucesso!")

    # preço médio por dia da semana
    # plt.figure(figsize=(12, 7))
    # preco_por_dia = df.groupby('day_of_week')['price'].mean()
    # sns.lineplot(x=preco_por_dia.index, y=preco_por_dia.values, marker='o')
    # plt.title('Preço Médio da Corrida por Dia da Semana', fontsize=16)
    # plt.xlabel('Dia da Semana', fontsize=12)
    # plt.ylabel('Preço Médio (em $)', fontsize=12)
    # # Traduzimos os números dos dias (0-6) para nomes legíveis
    # plt.xticks(ticks=range(7), labels=['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo'])
    # plt.grid(True)
    # plt.savefig('./imgs/dia_preco.png')
    # print("Gráfico './imgs/dia_preco.png' salvo com sucesso!")

    # cab_type e name - variaveis categoricas em numericas
    print("\nCriando novas features a partir do cab_type (uber ou lyft) e name (tipo de serviço)...")

    # seleção de colunas que serão transformadas
    #colunas_categoricas = ['cab_type', 'name']
    # fazer junto assim bugou o prefix no .get_dummies()

    # Aplicar dummies para 'cab_type' sem prefixo
    dummies_cab_type = pd.get_dummies(df['cab_type'])
    
    # Aplicar dummies para 'name' sem prefixo
    dummies_name = pd.get_dummies(df['name'])

    # função 'get_dummies' cria novas colunas
    # O 'prefix' ajuda a manter os nomes das colunas organizados (ex: cab_type_Lyft) - mas aqui não precisa
    #df_dummies = pd.get_dummies(df[colunas_categoricas])
    
    # uniao dessas novas colunas (dummies) ao DataFrame principal
    #df = pd.concat([df, df_dummies], axis=1)
    df = pd.concat([df, dummies_cab_type, dummies_name], axis=1)
    
    # Como as informações agora estão nas colunas de 0s e 1s, podemos remover as colunas de texto originais, poupa processamento para a modelagem
    #df.drop(columns=colunas_categoricas, inplace=True)
    df.drop(columns=['cab_type', 'name'], inplace=True)
    
    print("Colunas categóricas transformadas com sucesso!")
    # print("\nVisualização do DataFrame com as novas colunas (rolar para a direita para ver):")
    
    # # Usamos .info() para ver a lista completa de novas colunas
    # df.info()

    # print("\nAmostra dos dados com as novas colunas:")
    # print(df.head())

    # seleção de features para modelagem
    print("\nSelecionando as colunas finais para o treinamento do modelo...")

    # nossos features (X)
    # Inclui as numéricas, as de tempo e todas as novas colunas 'dummy'
    colunas_features = ['distance', 'surge_multiplier', 'hour', 'day_of_week', 'month'] + list(dummies_cab_type.columns) + list(dummies_name.columns) # 
    # adiciona as colunas criadas pelos .get_dummies()

    # coluna para previsao (Y)
    coluna_alvo = 'price'

    # Criando os DataFrames finais X e y
    X = df[colunas_features]
    y = df[coluna_alvo]

    print("DataFrames X e y criados com sucesso!")
    print(f"Nosso modelo usará {len(X.columns)} features para prever o preço.")
    print("Amostra das features (X):")
    print(X.head())

    # salvando df para modelagem em outro .py
    print("\nSalvando os DataFrames X e y para a próxima etapa...")

    # Criar uma pasta para os dados processados, se ela não existir
    pasta_dados_proc = atual_path / 'data_processed'
    pasta_dados_proc.mkdir(exist_ok=True)

    # salvando df via pickle (mais rápido e mantém tipos de dados)
    X.to_pickle(pasta_dados_proc / 'X_final.pkl')
    y.to_pickle(pasta_dados_proc / 'y_final.pkl')

    print(f"DataFrames salvos com sucesso na pasta: {pasta_dados_proc}")


except FileNotFoundError:
    print(f"ERRO: Arquivo não encontrado em '{csv_path}'.")
except Exception as e:
    print(f"Ocorreu um erro inesperado: {e}")