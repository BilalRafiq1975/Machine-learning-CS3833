import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from tabulate import tabulate

class CompensationAnalysis:
    def __init__(self, filename):
        self.df = pd.read_csv(filename)
        self.target_feature = 'compensation'

    def describe_dataset(self):
        description_table = tabulate(self.df.describe(), headers='keys', tablefmt='grid')
        print("")
        print("----------------------------------------------------")
        print("Dataset Description:")
        print(description_table)

    def plot_scatter_plots(self):
        sns.pairplot(self.df)
        plt.show()

    def split_dataset(self):
        X = self.df.drop(columns=[self.target_feature])
        y = self.df[self.target_feature]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def train_linear_regression(self):
        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)

    def predict_and_plot(self):
        y_pred = self.model.predict(self.X_test)
        plt.scatter(self.y_test, y_pred)
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title('True vs Predicted Values')
        plt.show()

        # Plotting residuals
        residuals = self.y_test - y_pred
        plt.scatter(self.y_test, residuals)
        plt.xlabel('True Values')
        plt.ylabel('Residuals')
        plt.axhline(y=0, color='r', linestyle='-')
        plt.title('Residual Plot')
        plt.show()

        # Calculating Mean Absolute Error
        mae = mean_absolute_error(self.y_test, y_pred)
        print("Mean Absolute Error:", mae)

    def print_r_squared_score(self):
        self.y_pred = self.model.predict(self.X_test)
        r_squared = r2_score(self.y_test, self.y_pred)
        print("----------------------------------------------------")
        print("R2-Score:", r_squared)
        print("")

    def analyze(self):
        self.describe_dataset()
        self.plot_scatter_plots()
        self.split_dataset()
        self.train_linear_regression()
        self.predict_and_plot()
        self.print_r_squared_score()

if __name__ == "__main__":
    # Creating an instance of the CompensationAnalysis class and analyzing the dataset
    analysis = CompensationAnalysis('compensation.csv')
    analysis.analyze()
