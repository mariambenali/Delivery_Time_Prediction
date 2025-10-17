import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score,mean_absolute_error




#load data
def load_file():
    df = pd.read_csv("data.csv")
    return df

print(load_file())


def data_cleaning(df):
    #remplir les NaN des colonnes num avec la moyenne
    df.fillna(df.mean(numeric_only=True), inplace=True)
    #remplir les NaN des colonnes catégoriques avec la valeur la plus
    for col in df.select_dtypes(include=['object', 'category']).columns:
        #df[col].fillna(df[col].mode()[0], inplace=True)
        df[col] = df[col].fillna(df[col].mode()[0])

    x = df.drop('Delivery_Time_min', axis=1)
    y = df["Delivery_Time_min"]
    return x ,y
df = load_file()              
x, y = data_cleaning(df)
print("x:", x.shape)
print("y:", y.shape)


def preprocessing(x):

    categorical_cols = x.select_dtypes(include=['object']).columns
    numerical_cols = x.select_dtypes(include=['int64', 'float64']).columns

    preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop=None), categorical_cols)
    ])
    return preprocessor
df = load_file()
x, y = data_cleaning(df)
preprocessor = preprocessing(x)
print(preprocessor)


def pipline(preprocessor):
    models = {
    "RandomForestRegressor": RandomForestRegressor(),
    "SVR": SVR()
}
    parameter_grid = {
    RandomForestRegressor: {
        'model__n_estimators': [100, 200, 500],
        'model__max_depth': [5, 10, 20],
        'model__random_state' :[42]
    },
    SVR: {
        'model__kernel': ['linear', 'rbf'],
        'model__gamma': ['scale', 'auto'],
        'model__C': [0.1, 1, 10]
    }
}
    KF = KFold(n_splits=5, shuffle=True, random_state=42)

    results = {}

    for name, model in models.items():
        print(f"🔹 Training model: {name}")
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('select', SelectKBest(score_func=f_regression, k=5)),
            ('model', model)  
        ])
        modeltype = type(model)

        grid_search = GridSearchCV(
            estimator= pipeline,
            param_grid=parameter_grid[modeltype],
            cv= KF,
            scoring= 'r2',
            n_jobs= 1
        )
        grid_search.fit(x, y) 
        results[name] = grid_search.best_estimator_
        best_model = grid_search.best_estimator_
        y_predict= best_model.predict(x)
        MAE = mean_absolute_error(y,y_predict)
        print(f"✅ Best R² for {name}: {grid_search.best_score_:.3f}")
        print(f"✅ {name} - MAE score with best estimator: {MAE:.3f}")
        print("-------")

    return results


trained_model = pipline(preprocessor)
print(trained_model)

