import os
import pickle

import optuna
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Carga de datos y split
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# funcion objetivo
def objective(trial):
    # Espacio de búsqueda
    n_estimators = trial.suggest_int("n_estimators", 10, 200)
    max_depth    = trial.suggest_int("max_depth", 2, 20)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    min_samples_leaf  = trial.suggest_int("min_samples_leaf", 1, 4)
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2"])
    
    # Pipeline: escalador + clasificador
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    # Validación cruzada (CV) con escalado incluido en cada fold
    scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=3,
        scoring="accuracy",
        n_jobs=-1
    )
    return scores.mean()

# 3) Creación y ejecución del Study
study = optuna.create_study(
    direction="maximize",
    study_name="rf_iris_tuning_with_scaling"
)
study.optimize(objective, n_trials=2, timeout=300)

# 4) Resultados
print("Trials completados:", len(study.trials))
print("Mejor accuracy:", study.best_value)
print("Mejores parámetros:", study.best_params)

# 5) Reentrenar el Pipeline completo sobre todo el set de entrenamiento
best_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(**study.best_params, random_state=42, n_jobs=-1))
])
best_pipeline.fit(X_train, y_train)

# 6) Guardado del modelo (Pipeline) en pickle
os.makedirs("models", exist_ok=True)
pkl_path = "models/iris_rf_optuna_scaled.pkl"
with open(pkl_path, "wb") as f:
    pickle.dump(best_pipeline, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Modelo óptimo (con escalado) guardado en {pkl_path}")
