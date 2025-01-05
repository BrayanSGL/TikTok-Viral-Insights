import pandas as pd

class DataProcessing:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        print("Objeto de DataProcessing creado")
        self.data = None
        
    def load_data(self):
        # Load data
        try:
            self.data = pd.read_csv(self.dataset_path)
            print("Datos cargados")
        except Exception as e:
            print(f"Error: {e}")

    def dimensions(self):
        print(f"Numero de filas: {self.data.shape[0]}")
        print(f"Numero de columnas: {self.data.shape[1]}")
        
    def show_head(self):
        print(self.data.head())
        
    def clean_data(self):
        # Drop rows with missing values
        self.data.dropna(inplace=True)
        print("Datos limpios, con valores nulos eliminados")
        
        