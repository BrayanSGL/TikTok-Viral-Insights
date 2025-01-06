import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import TreeSearch, BayesianEstimator


class ModelBuilder:
    def __init__(self, dataset_path):
        self.model = BayesianNetwork()
        self.dataset_path = dataset_path
        self.class_vaiable = 'video_view_count';
        self.tan_model = None
        
        
    def load_data(self):
        self.data = pd.read_csv(self.dataset_path)
        
    def build_model(self):
        # Initialize TreeSearch and fit it to the data
        est = TreeSearch(self.data, root_node=self.class_vaiable)
        self.tan_model = est.estimate(estimator_type="tan", class_node=self.class_vaiable)
        self.tan_model.fit_node_states(self.data, estimator=BayesianEstimator, prior_type="BDeu")
        
    def plot_model(self):
        # Plot the model
        pos = nx.spring_layout(self.tan_model)
        nx.draw(self.tan_model, pos, with_labels=True, node_size=3000, node_color="skyblue", font_size=10, font_weight="bold", font_color="black", edge_color="gray", linewidths=1, alpha=0.9, arrowsize=20, arrowstyle="->", connectionstyle="arc3,rad=0.1", labels={node: node for node in self.tan_model.nodes()})
        plt.title("TAN model")
        plt.show()
        
    def show_model(self):
        print(self.tan_model)
        
    def save_model(self):
        self.tan_model.to_file('model/model.bif')
        
