import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns #sirve para hacer graficos mas bonitos

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
        
    def get_atributes(self):
        return self.data.columns.tolist() #'#', 'claim_status', 'video_id', 'video_duration_sec', 'video_transcription_text', 'verified_status', 'author_ban_status', 'video_view_count', 'video_like_count', 'video_share_count', 'video_download_count', 'video_comment_count']
    
    def delite_atributes(self, list_atributes):
        self.data.drop(columns=list_atributes, inplace=True)
        print(f"Columnas {list_atributes} eliminadas")
        
    def encode_verified_status(self):
        self.data['verified_status'] = self.data['verified_status'].map({'verified': 1, 'not verified': 0})
        print("Columna 'verified_status' codificada")
        
    def save_dataset(self, path):
        self.data.to_csv(path, index=False)
        print(f"Datos guardados en {path}")
        
    def describe(self):
        print(self.data.describe())
        #guadar los resultados de la descripcion en un csv
        self.data.describe().to_csv('data/processed/analytics/description.csv')
        
    def plot_correlation(self):
        plt.figure(figsize=(12, 8))
        sns.heatmap(self.data.corr(), annot=True, cmap='coolwarm')
        plt.title("Correlación de atributos")
        #guardar imagen en formato png
        plt.savefig('data/processed/analytics/correlation.png')
        
        # guardar los resultados de la correlacion en un csv
        self.data.corr().to_csv('data/processed/analytics/correlation.csv')
        
    def plot_histogram(self):
        self.data.hist(figsize=(12, 8))
        plt.suptitle("Histograma de atributos")
        plt.savefig('data/processed/analytics/histogram.png')
                  
   