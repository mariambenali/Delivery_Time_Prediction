
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from Pipline import load_file, data_cleaning, preprocessing, pipline


def test_load_file():
    df = load_file()
    assert df is not None
    assert not df.empty


def test_data_cleaning():
    df= load_file()
    x,y = data_cleaning(df)
    assert x is not None
    assert y is not None
    assert not x.empty
    assert len(y)>0
    assert x.shape[0] == y.shape[0]


def test_preprocessing ():
    df =load_file()
    x,y = data_cleaning(df)
    preprocessor= preprocessing(x)
    assert isinstance(preprocessor,ColumnTransformer)



def test_pipeline():
    df = load_file()
    x,y = data_cleaning(df)
    preprocessor = preprocessing(x)
    trained_models = pipline(preprocessor)
    assert "SVR" in trained_models
    assert "RandomForestRegressor" in trained_models






