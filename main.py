from src.data_processing import DataProcessing
from src.model_builder import ModelBuilder

def main():

    dataset_path_raw = "data/raw/tiktok_dataset.csv"  # Ruta del dataset sin procesar
    dataset_parh_processed = "data/processed/tiktok_dataset.csv"  # Ruta del dataset procesado
    
    print("llamando al procesamiento de datos")
    data = DataProcessing(dataset_path_raw)  # Inicializa el procesamiento de datos
    data.load_data()  # Carga los datos
    data.clean_data()  # Limpia los datos
    data.delite_atributes(['#', 'claim_status', 'video_id', 'video_transcription_text', 'author_ban_status'])  # Elimina columnas
    data.encode_verified_status()  # Codifica el estado verificado
    data.save_dataset(dataset_parh_processed)  # Guarda el dataset procesado
    data.describe()  # Muestra estadisticas del dataset
    data.plot_correlation()  # Grafica la correlacion de los atributos
    data.plot_histogram()  # Grafica el histograma de los atributos
    
    print("llamando al modelo")
    model = ModelBuilder(dataset_parh_processed)  # Inicializa el modelo
    model.load_data()  # Carga los datos
    #model.split_data()  # Divide los datos en entrenamiento y prueba
    model.build_model()  # Construye el modelo
    model.plot_model()  # Grafica el modelo
    

if __name__ == "__main__":
    main()