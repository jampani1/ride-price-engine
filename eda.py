#variavel alvo y - preço que sera previsto
#variavel x - features para ajudar na previsao (distancia, tipo de corrida, hora?, dia da semana, clima, eventos especiais, localizacao

import pandas as pd
import os

original_file = os.path.join(os.path.dirname(__file__), 'data_source', 'rideshare_kaggle.csv')

try:
    # Read CSV file
    print("Reading CSV file...")
    df = pd.read_csv(original_file, encoding='latin1')

    print("\nDataset shape:", df.shape)
    print("\nFirst 5 records:")
    print(df.head())

    print("\nDataset Info:")
    print(df.info())

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nBasic Statistics:")
    print(df.describe())

except FileNotFoundError as e:
    print(f"File error: {str(e)}")
    print("Please check if the file was downloaded correctly")