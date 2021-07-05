import pandas as pd
import numpy as np
import unicodedata
import re
import json
import nltk

from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, RFE, f_regression
from sklearn.preprocessing import MinMaxScaler

#------------------------------------------------------------------------------------------------------

def split_dataframe(df, stratify_by=None, rand=1414, test_size=.2, validate_size=.3):
    
    if stratify_by == None:
        train, test = train_test_split(df, test_size=test_size, random_state=rand)
        train, validate = train_test_split(train, test_size=validate_size, random_state=rand)
    else:
        train, test = train_test_split(df, test_size=test_size, random_state=rand, stratify=df[stratify_by])
        train, validate = train_test_split(train, test_size=validate_size, random_state=rand, stratify=train[stratify_by])

    return train, validate, test

#------------------------------------------------------------------------------------------------------

def split_dataframe_continuous_target(dframe, target, bins=5, rand=1414, test_size=.2, validate_size=.3):
    
    df = dframe.copy()
    binned_y = pd.cut(df[target], bins=bins, labels=list(range(bins)))
    df["bins"] = binned_y

    train_validate, test = train_test_split(df, stratify=df["bins"], test_size=test_size, random_state=rand)
    train, validate = train_test_split(train_validate, stratify=train_validate["bins"], test_size=validate_size, random_state=rand)

    train = train.drop(columns=["bins"])
    validate = validate.drop(columns=["bins"])
    test = test.drop(columns=["bins"])
    
    return train, validate, test