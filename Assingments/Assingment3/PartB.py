from sklearn.model_selection import cross_val_score, KFold
from PartA import CompensationAnalysis
from tabulate import tabulate

class CompensationAnalysisWithKFold(CompensationAnalysis):
    def perform_kfold_validation(self, splits=[3, 5, 10]):
        for split in splits:
            kf = KFold(n_splits=split, shuffle=True, random_state=42)
            scores = cross_val_score(self.model, self.X_train, self.y_train, cv=kf)
            print("")
            print("+-----------+-----------+-----------+-----------+-----------+")
            print(f"Validation Scores for {split} splits:")
            print(tabulate([scores], headers=[f"Split {i+1}" for i in range(split)], tablefmt="grid"))
            print(f"Mean Validation Score for {split} splits:", scores.mean())
            print("+-----------+-----------+-----------+-----------+-----------+")
            print("")
# Creating an instance of the CompensationAnalysisWithKFold class and analyzing the dataset
analysis_with_kfold = CompensationAnalysisWithKFold('compensation.csv')
analysis_with_kfold.split_dataset()  # Splitting dataset for training
analysis_with_kfold.train_linear_regression()  # Training the model
analysis_with_kfold.perform_kfold_validation()  # Performing K-fold validation
