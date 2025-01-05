from src.data_processing import DataProcessing

def main():

    dataset_path = "data/raw/tiktok_dataset.csv"
    data = DataProcessing(dataset_path)
    data.load_data()
    data.dimensions()
    data.clean_data()
    data.dimensions()

if __name__ == "__main__":
    main()