import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import TreeSearch, BayesianEstimator
from pgmpy.readwrite import BIFWriter
import traceback


class ModelBuilder:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.class_vaiable = 'video_view_count'
        self.tan_model = None
        print("Modelo inicializado")
        
    def load_data(self):
        self.data = pd.read_csv(self.dataset_path)
        print("Datos cargados")
    
    def split_data(self):
        # Divide los datos en entrenamiento y prueba
        self.train_data = self.data.sample(frac=0.8, random_state=200)
        self.test_data = self.data.drop(self.train_data.index)
        print("Datos divididos")
    
    def build_model(self):
        try:
            # Selecciona un nodo raíz diferente al nodo clase
            root_node = self.data.columns.difference([self.class_vaiable])[0]
            print(f"Nodo raíz seleccionado: {root_node}")
            
            # Construye el modelo TAN
            est = TreeSearch(self.data, root_node=root_node)
            dag = est.estimate(estimator_type="tan", class_node=self.class_vaiable)
            self.tan_model = BayesianNetwork(dag.edges())  # Convierte el grafo en una red bayesiana
            
            # Ajusta los parámetros del modelo
            self.tan_model.fit(self.data, estimator=BayesianEstimator, prior_type="BDeu")
            print("Modelo TAN ajustado y construido con éxito")
        except Exception as e:
            print(f"Error al construir el modelo: {e}")
            traceback.print_exc()
    
    def plot_model(self):
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.tan_model)
        nx.draw(
            self.tan_model,
            pos,
            with_labels=True,
            node_size=3000,
            node_color="skyblue",
            font_size=10,
            font_weight="bold",
            edge_color="gray",
            alpha=0.9,
            arrowsize=20,
            arrowstyle="->",
        )
        plt.title("Modelo TAN")
        plt.show()
        
    def show_model(self):
        print(self.tan_model.edges())
        
    def save_model(self):
        writer = BIFWriter(self.tan_model)
        writer.write_bif('model/model.bif')
        print("Modelo guardado en model/model.bif")

        
