import os

def download_datasets():
    print("Attempting to download datasets from Kaggle...")
    
    # Check if kaggle is installed
    try:
        import kaggle
    except ImportError:
        print("Error: 'kaggle' library not installed. Run 'pip install kaggle'")
        return

    datasets = [
        "openfoodfacts/world-food-facts",
        "yashkaggle27/nutrition-dataset-for-healthy-food-prediction"
    ]
    
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    for dataset in datasets:
        print(f"Downloading {dataset}...")
        try:
            # This requires ~/.kaggle/kaggle.json to be present
            kaggle.api.dataset_download_files(dataset, path=data_dir, unzip=True)
            print(f"Successfully downloaded and unzipped {dataset}")
        except Exception as e:
            print(f"Failed to download {dataset}: {e}")
            print("Make sure you have your Kaggle API key (kaggle.json) in ~/.kaggle/")

if __name__ == "__main__":
    download_datasets()
