# src/train_model.py

import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from preprocessing import load_data, get_features_target_preprocessor
from sklearn.metrics import accuracy_score

def main():
    df = load_data("data/HR_Employee_Attrition-1.csv")
    X, y, preprocessor = get_features_target_preprocessor(df)

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10, 20],
        'classifier__min_samples_split': [2, 5]
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(x_train, y_train)

    print("Best Parameters:", grid_search.best_params_)
    print("Best CV Accuracy:", grid_search.best_score_)

    y_pred = grid_search.predict(x_test)
    print("Test Accuracy:", accuracy_score(y_test, y_pred))

    with open("../models/pipeline.pkl", "wb") as f:
        pickle.dump(grid_search.best_estimator_, f)
    print("Saved pipeline to models/pipeline.pkl")

if __name__ == "__main__":
    main()
