import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class DataAnalyzer:
    def __init__(self, data):
        """Inicializa la clase con un DataFrame."""
        self.data = data

    def generate_descriptive_statistics(self):
        """Genera estadísticas descriptivas de las columnas numéricas y categóricas."""
        if self.data.empty:
            raise ValueError("El DataFrame está vacío.")

        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        categorical_columns = self.data.select_dtypes(include=['object', 'category']).columns

        # Estadísticas para columnas numéricas
        stats = self.data[numeric_columns].describe() if numeric_columns.size > 0 else None

        # Resumen categórico con proporciones
        categorical_summary = (
            self.data[categorical_columns]
            .apply(lambda x: x.value_counts(normalize=True) * 100)
            .T
        ) if categorical_columns.size > 0 else None

        return stats, categorical_summary

    def plot_numeric_distributions(self):
        """Genera histogramas y KDE para columnas numéricas."""
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        if numeric_columns.size == 0:
            print("No hay columnas numéricas para graficar.")
            return

        for col in numeric_columns:
            sns.histplot(self.data[col], kde=True, color='blue')
            plt.title(f'Distribución de {col}')
            plt.xlabel(col)
            plt.ylabel('Frecuencia')
            plt.show()

    def plot_correlation_matrix(self):
        """Genera un mapa de calor para mostrar correlaciones entre variables numéricas."""
        numeric_columns = self.data.select_dtypes(include=[np.number])
        if numeric_columns.empty:
            print("No hay columnas numéricas para calcular correlaciones.")
            return

        correlation_matrix = numeric_columns.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Matriz de correlación')
        plt.show()

    def compare_categorical_impact(self, categorical_col, numeric_col):
        """Compara cómo una columna categórica impacta una métrica numérica."""
        if categorical_col not in self.data.columns or numeric_col not in self.data.columns:
            print(f"{categorical_col} o {numeric_col} no existen en los datos.")
            return

        sns.boxplot(x=self.data[categorical_col], y=self.data[numeric_col])
        plt.title(f'Impacto de {categorical_col} en {numeric_col}')
        plt.xlabel(categorical_col)
        plt.ylabel(numeric_col)
        plt.xticks(rotation=45)
        plt.show()

    def plot_scatter(self, col_x, col_y):
        """Genera un diagrama de dispersión para dos columnas numéricas."""
        if col_x not in self.data.columns or col_y not in self.data.columns:
            print(f"{col_x} o {col_y} no existen en los datos.")
            return

        sns.scatterplot(x=self.data[col_x], y=self.data[col_y])
        plt.title(f'Dispersión entre {col_x} y {col_y}')
        plt.xlabel(col_x)
        plt.ylabel(col_y)
        plt.show()

    def prepare_for_tan_model(self):
        """Prepara los datos para el modelo TAN visualizando relaciones condicionales."""
        numeric_columns = self.data.select_dtypes(include=[np.number])
        if numeric_columns.empty:
            print("No hay columnas numéricas para preparar el modelo TAN.")
            return

        correlation_matrix = numeric_columns.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='viridis', fmt='.2f')
        plt.title('Matriz de correlación para el modelo TAN')
        plt.show()

    def calculate_conditional_probabilities(self, col1, col2):
        """Calcula la tabla de probabilidad condicional entre dos columnas categóricas."""
        if col1 not in self.data.columns or col2 not in self.data.columns:
            print(f"{col1} o {col2} no existen en los datos.")
            return

        contingency_table = pd.crosstab(self.data[col1], self.data[col2], normalize='columns')
        print(f"Probabilidades condicionales entre {col1} y {col2}:")
        print(contingency_table)
        return contingency_table


# Ejemplo de uso
if __name__ == "__main__":
    # Ruta robusta para el archivo
    file_path = os.path.join(os.path.dirname(__file__), "../data/processed/tiktok_dataset.csv")

    try:
        # Cargar datos de ejemplo
        data = pd.read_csv(file_path)

        # Instanciar la clase
        analyzer = DataAnalyzer(data)

        # Generar estadísticas descriptivas
        stats, cat_summary = analyzer.generate_descriptive_statistics()
        print("Estadísticas numéricas:")
        print(stats)
        print("\nResumen categórico con proporciones:")
        print(cat_summary)

        # Graficar distribuciones
        analyzer.plot_numeric_distributions()

        # Mostrar matriz de correlación
        analyzer.plot_correlation_matrix()

        # Comparar impacto de columnas categóricas en métricas
        analyzer.compare_categorical_impact(categorical_col='autor_verificado', numeric_col='likes')

        # Generar un gráfico de dispersión
        analyzer.plot_scatter(col_x='duracion_video', col_y='likes')

        # Preparar datos para el modelo TAN
        analyzer.prepare_for_tan_model()

        # Calcular probabilidades condicionales
        analyzer.calculate_conditional_probabilities(col1='autor_verificado', col2='estado_reclamado')

    except FileNotFoundError:
        print(f"El archivo no se encontró: {file_path}")
    except Exception as e:
        print(f"Error inesperado: {e}")
