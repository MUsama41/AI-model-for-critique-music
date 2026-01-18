import pickle
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from src.utils.db_helper import DatabaseClient

class ModelTrainer:
    def __init__(self, model_path='model.pkl'):
        self.model_path = model_path
        self.db = DatabaseClient()

    def train_random_forest(self, X_train, y_train):
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'bootstrap': [True]
        }
        grid_search = GridSearchCV(
            estimator=RandomForestRegressor(random_state=42),
            param_grid=param_grid,
            cv=3, n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_

    def retrain(self):
        if not self.db.migrate_retrain_data():
            return False
            
        df = self.db.get_training_data()
        if df.empty: return False
        
        X = df.drop(columns=['popularity'])
        y = df['popularity']
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.1, random_state=42)
        
        model = self.train_random_forest(X_train, y_train)
        with open(self.model_path, 'wb') as f:
            pickle.dump(model, f)
            
        return True
