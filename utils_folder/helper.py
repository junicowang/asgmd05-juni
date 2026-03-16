import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

CATEGORICAL_FEATURES = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side', 'Age_group']
NUMERICAL_FEATURES   = [
    'Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
    'Cabin_num', 'Group_size', 'Solo', 'Family_size', 'TotalSpending',
    'HasSpending', 'NoSpending', 'Age_missing', 'CryoSleep_missing',
    'RoomService_ratio', 'FoodCourt_ratio', 'ShoppingMall_ratio',
    'Spa_ratio', 'VRDeck_ratio'
]

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y= None):
        return self
    
    def transform(self, X):
        df = X.copy()

        # Cabin → Deck, Cabin_num, Side
        df['Deck']      = df['Cabin'].apply(lambda x: x.split('/')[0] if pd.notna(x) else 'Unknown')
        df['Cabin_num'] = df['Cabin'].apply(lambda x: x.split('/')[1] if pd.notna(x) else -1).astype(float)
        df['Side']      = df['Cabin'].apply(lambda x: x.split('/')[2] if pd.notna(x) else 'Unknown')

        # Group features
        df['Group']      = df['PassengerId'].apply(lambda x: x.split('_')[0])
        df['Group_size'] = df.groupby('Group')['Group'].transform('count')
        df['Solo']       = (df['Group_size'] == 1).astype(int)

        # Family features
        df['FirstName']   = df['Name'].apply(lambda x: x.split()[0]  if pd.notna(x) else 'Unknown')
        df['LastName']    = df['Name'].apply(lambda x: x.split()[-1] if pd.notna(x) else 'Unknown')
        df['Family_size'] = df.groupby('LastName')['LastName'].transform('count')

        # Spending features
        spending_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        for col in spending_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        df['TotalSpending'] = df[spending_cols].sum(axis=1)
        df['HasSpending']   = (df['TotalSpending'] > 0).astype(int)
        df['NoSpending']    = (df['TotalSpending'] == 0).astype(int)
        for col in spending_cols:
            df[f'{col}_ratio'] = df[col] / (df['TotalSpending'] + 1)

        # Age features
        df['Age_group'] = pd.cut(
            df['Age'], bins=[0, 12, 18, 30, 50, 100],
            labels=['Child', 'Teen', 'Young_Adult', 'Adult', 'Senior']
        ).astype(str)
        df['Age_group'] = df['Age_group'].replace('nan', 'Unknown')

        # Missing indicators
        df['Age_missing']       = df['Age'].isna().astype(int)
        df['CryoSleep_missing'] = df['CryoSleep'].isna().astype(int)

        # Convert boolean → string untuk OrdinalEncoder
        df['CryoSleep'] = df['CryoSleep'].astype(str)
        df['VIP']       = df['VIP'].astype(str)

        return df[CATEGORICAL_FEATURES + NUMERICAL_FEATURES]

def build_pipeline(C=1.0, penalty='l2', solver='lbfgs', max_iter=1000):

    # Sub-pipeline untuk kolom kategorik
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
    ])

    # Sub-pipeline untuk kolom numerik
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
    ])

    # Gabungin pakai ColumnTransformer
    preprocessor = ColumnTransformer([
        ('cat', cat_pipeline, CATEGORICAL_FEATURES),
        ('num', num_pipeline, NUMERICAL_FEATURES),
    ])

    # Full pipeline: FeatureEngineer → Preprocessor → Model
    full_pipeline = Pipeline([
        ('feature_engineer', FeatureEngineer()),
        ('preprocessor', preprocessor),
        ('model', LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=max_iter, random_state=42)),
    ])

    return full_pipeline


