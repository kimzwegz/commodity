import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn import metrics

from sklearn import ensemble
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

class Feature:
    def __init__(self,df):
        self.df = df
        print('object instanciated')

    
def train(self, df=None, col_period='cal_yearperiod' , train_window=4 , test_window=1, test_gap = 0, expanding=False):
        df = self.df_load(df)
        test_train = []
        # test_train = {}

        periods = df[col_period].unique().tolist()
        periods.sort()

        if expanding == False:

            for i,j in enumerate(periods):
                if i < len(periods)-train_window-1:
                    train_beg = j
                    train_beg_idx = periods.index(train_beg)

                    train_end_idx = train_beg_idx+ train_window
                    train_end = periods[train_end_idx]
                    
                    test_beg_idx = train_end_idx+test_gap
                    test_end_idx = test_beg_idx+test_window

                    # train = periods[train_idx]
                    train_periods = periods[train_beg_idx: train_end_idx]
                    test_periods = periods[test_beg_idx: test_end_idx]

                    df_train = df.loc[df.cal_yearperiod.isin(train_periods)]
                    df_test = df.loc[df.cal_yearperiod.isin(test_periods)]

                    test_train.append((df_train, df_test))
