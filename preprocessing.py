import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import random
import re
import sys
import ktrain
from ktrain import text
from joblib import load

class preprocessing():

    def __init__(self):
        self.cat_enc = LabelEncoder()
        self.sub_cat_enc = LabelEncoder()

    def duplicate(self, df):
        less = (df['SUB-CATEGORIA'].value_counts() < 2)
        cats = []
        for i in range(len(less)):
            if (less[i]):
                cats.append(less.keys()[i])
        for i in range(len(df)):
            if (df['SUB-CATEGORIA'][i] in cats):
                df = df.append(df.iloc[i, :])
        return df

    def labelencoder(self, df, column, inverse = False):
        if(inverse == False):
            if (column == 'SUB-CATEGORIA'):
                return self.sub_cat_enc.fit_transform(df[column])
            elif (column == 'CATEGORIA'):
                return self.cat_enc.fit_transform(df[column])
            else:
                print("NÃO ENCONTRAMOS A COLUNA: " + column)
        else:
            if (column == 'SUB-CATEGORIA'):
                return self.sub_cat_enc.inverse_transform(df)
            elif (column == 'CATEGORIA'):
                return self.cat_enc.inverse_transform(df)
            else:
                print("NÃO ENCONTRAMOS A COLUNA: " + column)


