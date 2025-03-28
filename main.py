import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, PowerTransformer
from itertools import combinations
from sklearn.metrics import accuracy_score, classification_report, precision_recall_curve

class AirlineSatisfactionPreprocessor:
    def __init__(self):
        self.encoding_dict = {}
        self.label_encoders = {}
        self.onehot_encoder = OneHotEncoder(sparse_output=False, drop='first')
        self.power_transformer = PowerTransformer(method="yeo-johnson")
        
    def preprocess(self, df):
        # Drop unnecessary columns
        df = df.dropna()
        df = df.drop(columns=['Unnamed: 0', 'id'], errors='ignore')
        
        # Define column types
        cat_cols = ['Type of Travel', 'Customer Type', 'Class', 'satisfaction']
        oh_col = ['Gender']
        
        # Label Encoding
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le
            self.encoding_dict[col] = dict(zip(le.classes_, range(len(le.classes_))))
        
        # One-Hot Encoding
        encoded_data = self.onehot_encoder.fit_transform(df[oh_col])
        encoded_df = pd.DataFrame(
            encoded_data, 
            columns=self.onehot_encoder.get_feature_names_out(oh_col)
        )

        # Combine dataframes
        df = df.drop(columns=oh_col).reset_index(drop=True)
        df = pd.concat([df, encoded_df], axis=1)
        
        return df
    
    def transform_skewed_features(self, df):
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        skewed_cols = df[numerical_cols].apply(lambda x: x.skew()).abs()
        skewed_cols = skewed_cols[skewed_cols > 0.5].index
        
        # Apply Yeo-Johnson transformation
        df[skewed_cols] = self.power_transformer.fit_transform(df[skewed_cols])
        
        return df
    
    def age_group_transformation(self, df):
        bins = [7, 18, 30, 40, 50, 60, 70, 80, 85]
        labels = ['7-17', '18-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-85']
        df['age_group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
        df = df.drop(columns=['Age'])
        df = pd.get_dummies(df, columns=['age_group'], prefix='age')
        
        return df
    
    def feature_interaction(self, df):
        columns = ['Online boarding', 'Inflight wifi service', 'Inflight service', 'Type of Travel']
        new_columns = []
        
        for feature1, feature2 in combinations(columns, 2):
            if feature1 != 'satisfaction' and feature2 != 'satisfaction':
                interaction_feature_name = f"{feature1}_x_{feature2}"
                new_column = (df[feature1] + df[feature2])
                new_columns.append(pd.Series(new_column, name=interaction_feature_name))
        
        df = pd.concat([df] + new_columns, axis=1)
        return df
    
    def full_pipeline(self, df):
        df = self.preprocess(df)
        df = self.age_group_transformation(df)
        df = self.transform_skewed_features(df)
        df = self.feature_interaction(df)
        
        return df

class ModelEvaluator:
    @staticmethod
    def evaluate_model(model, X, y):
        y_pred = model.predict(X)
        print("Accuracy:", accuracy_score(y, y_pred))
        print(classification_report(y, y_pred))
    
    @staticmethod
    def plot_precision_recall_curve(y_test, y_pred_proba):
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, marker='.')
        plt.title('Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.show()

def main():
    test_df = pd.read_csv('data/test.csv')
    preprocessor = AirlineSatisfactionPreprocessor()
    
    # Preprocess data
    processed_df = preprocessor.full_pipeline(test_df)
    
    # Split features and target
    X = processed_df.drop(columns=['satisfaction'])
    y = processed_df['satisfaction']

    # Model Import
    model = joblib.load('models/cat_model.pkl')
    ModelEvaluator.evaluate_model(model, X, y)
    ModelEvaluator.plot_precision_recall_curve(y, model.predict_proba(X)[:, 1])

if __name__ == "__main__":
    main()