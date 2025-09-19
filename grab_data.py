import kaggle
import pandas as pd
import os

try:
    data_dir = os.path.join(os.path.dirname(__file__), 'data_source')
    os.makedirs(data_dir, exist_ok=True)
    
    
    # Test authentication
    print("Testing Kaggle authentication...")
    kaggle.api.authenticate()
    print("Authentication successful!")

    # Download dataset
    print("Downloading dataset...")
    kaggle.api.dataset_download_files(
        'brllrb/uber-and-lyft-dataset-boston-ma',
        path=data_dir,  # Changed to download directly to data_source
        unzip=True,
        force=True
    )
    
    # Define file paths to windows
    original_file = os.path.join(data_dir, 'rideshare_kaggle.csv')
    cleaned_file = os.path.join(data_dir, 'saved_rideshare_data.csv')
    
    # Verify file exists
    if not os.path.exists(original_file):
        raise FileNotFoundError(f"Dataset file not found at {original_file}")

    # Read CSV file
    print("Reading CSV file...")
    df = pd.read_csv(
        original_file,
        encoding='latin1'
    )
    
    print("\nDataset shape:", df.shape)
    print("\nFirst 5 records:")
    print(df.head())

    print("\nDataset Info:")
    print(df.info())
    
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    print("\nBasic Statistics:")
    print(df.describe())
    
    # Save processed data
    df.to_csv(cleaned_file, index=False)
    print(f"\nData saved to: {os.path.abspath(cleaned_file)}")

except FileNotFoundError as e:
    print(f"File error: {str(e)}")
    print("Please check if the file was downloaded correctly")
except Exception as e:
    print(f"Error: {str(e)}")
    print("\nTroubleshooting steps:")
    print("1. Check if kaggle.json exists in:", os.path.join(os.path.expanduser('~'), '.kaggle'))
    print("2. Try running 'kaggle datasets list' in terminal to test authentication")