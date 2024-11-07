#!/usr/bin/env python3

from __future__ import annotations
from typing import Optional, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


def read_csv(filename: str) -> pd.DataFrame:
    return pd.read_csv(filename, sep=',', encoding='latin-1')

train_and_val_data = read_csv('data/ds4420_kaggle_train_data.csv')
test_data = read_csv('data/ds4420_kaggle_test_data.csv')

def split_label(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return data.drop('Label', axis=1), data['Label']

X_train_and_val, y_train_and_val = split_label(train_and_val_data)
X_train, X_val, y_train, y_val = train_test_split(X_train_and_val,
                                                      y_train_and_val,
                                                      test_size=0.2,
                                                      random_state=1)

X_test = test_data

def evaluate_and_save_results(pipeline: Pipeline, model_num: int) -> None:
    
    pipeline.fit(X_train, y_train)
    
    y_train_pred = pipeline.predict(X_train)
    y_val_pred = pipeline.predict(X_val)
    y_test_pred = pipeline.predict(X_test)
    
    print(classification_report(y_val, y_val_pred))
    
    test_data_exportable = test_data.copy()
    test_data_exportable['Label'] = y_test_pred
    test_data_exportable = test_data_exportable[['ID', 'Label']]
    test_data_exportable.to_csv(f'outputs/model{model_num}.csv', index=False)



class LengthFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Converts a text feature into its length.
    """
    
    def __init__(self) -> None:
        BaseEstimator.__init__(self)
        TransformerMixin.__init__(self)
    
    def fit(self, X: pd.Series, y: Optional[Any]=None) -> LengthFeatureExtractor:
        return self
        
    def transform(self, X: pd.Series) -> pd.DataFrame:
        yelling_feature = X.apply(lambda text: len(text))
        return yelling_feature.values.reshape(-1, 1)

class CapsBinaryFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Converts a text feature into a binary feature where 1 denotes that at least one word in the text
    is completely capitalized, which is assumed to be yelling.
    """
    
    def __init__(self) -> None:
        BaseEstimator.__init__(self)
        TransformerMixin.__init__(self)
        
    def __text_contains_caps_binary(self, text: str) -> int:
        return int(any(word.isupper() for word in text.split(' ')))
    
    def fit(self, X: pd.Series, y: Optional[Any]=None) -> CapsBinaryFeatureExtractor:
        return self
        
    def transform(self, X: pd.Series) -> pd.DataFrame:
        yelling_feature = X.apply(self.__text_contains_caps_binary)
        return yelling_feature.values.reshape(-1, 1)

class CapsCountFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Converts a text feature into a numeric feature count of the number of words in the text that are
    completely capitalized, which is assumed to be yelling.
    """
    
    def __init__(self) -> None:
        BaseEstimator.__init__(self)
        TransformerMixin.__init__(self)
    
    def __text_caps_count(self, text: str) -> int:
        return int(sum(1 for word in text.split(' ') if word.isupper()))
    
    def fit(self, X: pd.Series, y: Optional[Any]=None) -> CapsCountFeatureExtractor:
        return self
    
    def transform(self, X: pd.Series) -> pd.DataFrame:
        yelling_feature = X.apply(self.__text_caps_count)
        return yelling_feature.values.reshape(-1, 1)

class CharBinaryFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Converts a text feature into binary feature where 1 indicates the text containst at least one of
    the given character.
    """

    char: str
    
    def __init__(self, char: str) -> None:
        BaseEstimator.__init__(self)
        TransformerMixin.__init__(self)

        if len(char) != 1:
            raise ValueError('Character must be a string of length 1.')
        
        self.char = char
    
    def fit(self, X: pd.Series, y: Optional[Any]=None) -> ExclamationBinaryFeatureExtractor:
        return self
    
    def transform(self, X: pd.Series) -> pd.DataFrame:
        yelling_feature = X.apply(lambda text: 0 + (self.char in text))
        return yelling_feature.values.reshape(-1, 1)

class CharCountFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Converts a text feature into a numeric feature count of the number of exclamation marks, which is
    assumed to be yelling.
    """

    char: str
    
    def __init__(self, char: str) -> None:
        BaseEstimator.__init__(self)
        TransformerMixin.__init__(self)
    
        if len(char) != 1:
            raise ValueError('Character must be a string of length 1.')
        
        self.char = char

    def fit(self, X: pd.Series, y: Optional[Any]=None) -> ExclamationCountFeatureExtractor:
        return self
    
    def transform(self, X: pd.Series) -> pd.DataFrame:
        yelling_feature = X.apply(lambda text: text.count(self.char))
        return yelling_feature.values.reshape(-1, 1)

class SwearingFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Converts a text feature into a binary feature where 1 denotes swearing occurs, which is usually
    seen in text as a series of 2 or more asterisks, e.g. ****.
    """
    
    def __init__(self) -> None:
        BaseEstimator.__init__(self)
        TransformerMixin.__init__(self)
    
    def fit(self, X: pd.Series, y: Optional[Any]=None) -> SwearingFeatureExtractor:
        return self
    
    def transform(self, X: pd.Series) -> pd.DataFrame:
        yelling_feature = X.apply(lambda text: 0 + ('**' in text))
        return yelling_feature.values.reshape(-1, 1)




text_vectorizer = CountVectorizer(
    stop_words='english',
)
    
preprocessor = ColumnTransformer(
    transformers=[
        ('text', text_vectorizer, 'Text'),
        ('numeric', StandardScaler(), ['User_Age', 'Time_of_Post', 'Population_Density']),
        ('capitalization', CapsCountFeatureExtractor(), 'Text'),
    ]
)
    
pipeline = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestClassifier(n_estimators=225, random_state=1)),
    ]
)

param_grid = {
    'n_estimators': np.arange(221, 230, 1)
}

print('Beginning gridsearch...')

grid_search = GridSearchCV(RandomForestClassifier(random_state=1), param_grid, cv=3, scoring='f1')
grid_search.fit(preprocessor.fit_transform(X_train_and_val), y_train_and_val)

print(grid_search.best_params_)
