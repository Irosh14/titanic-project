# train.py  - a very simple trainer that saves model.pkl
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

print("Loading data...")
df = pd.read_csv("data/train.csv")   # must be at data/train.csv

features = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
X = df[features]
y = df['Survived']

# preprocessing
num_features = ['Age','SibSp','Parch','Fare']
num_transformer = Pipeline([('imputer', SimpleImputer(strategy='median')),
                            ('scaler', StandardScaler())])
cat_features = ['Sex','Embarked','Pclass']
cat_transformer = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                            ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer([('num', num_transformer, num_features),
                                  ('cat', cat_transformer, cat_features)])

print("Setting up models...")
pipe_rf = Pipeline([('pre', preprocessor), ('clf', RandomForestClassifier(random_state=42))])
pipe_log = Pipeline([('pre', preprocessor), ('clf', LogisticRegression(max_iter=1000))])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("Evaluating RandomForest... (this may take a moment)")
rf_scores = cross_val_score(pipe_rf, X, y, cv=cv, scoring='accuracy')
print("RandomForest accuracy (mean):", rf_scores.mean().round(3))

print("Evaluating LogisticRegression...")
log_scores = cross_val_score(pipe_log, X, y, cv=cv, scoring='accuracy')
print("LogisticRegression accuracy (mean):", log_scores.mean().round(3))

# pick best
if rf_scores.mean() >= log_scores.mean():
    best = pipe_rf
    name = "RandomForest"
else:
    best = pipe_log
    name = "LogisticRegression"

print("Training final model:", name)
best.fit(X, y)
joblib.dump(best, "model.pkl")
print("Saved model as model.pkl")
