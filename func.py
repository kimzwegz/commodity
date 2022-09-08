import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn import metrics

from sklearn import ensemble
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

class Model:
    def __init__(self,df):
        self.df = df
        print('object instanciated')

    def train(self, df=None, col_period=None , train_window=4 , test_window=1, test_gap = 0, expanding=False):
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

                    df_train = df.loc[df[col_period].isin(train_periods)]
                    df_test = df.loc[df[col_period].isin(test_periods)]

                    test_train.append((df_train, df_test))
                    # test_train[i] = {"train": df_train , "test": df_test}

        else:
            for i,j in enumerate(periods):
                if i < len(periods)-train_window-1:
                    train_beg = periods[0]
                    train_beg_idx = periods.index(train_beg)

                    train_end_idx = train_beg_idx+ train_window+i
                    train_end = periods[train_end_idx]
                    
                    test_beg_idx = train_end_idx+test_gap
                    test_end_idx = test_beg_idx+test_window

                    # train = periods[train_idx]
                    train_periods = periods[train_beg_idx: train_end_idx]
                    test_periods = periods[test_beg_idx: test_end_idx]

                    df_train = df.loc[df[col_period].isin(train_periods)]
                    df_test = df.loc[df[col_period].isin(test_periods)]

                    test_train.append((df_train, df_test))
                    # test_train[i] = {"train": df_train , "test": df_test}
      
        return test_train

    def skpredict(self, df_train, df_test, skmodel, cols_x, cols_y, printstat=True):
        regr = skmodel
        X_train = df_train[cols_x]
        Y_train = df_train[cols_y]
        
        X_test = df_test[cols_x]
        Y_test = df_test[cols_y]
        
        regr.fit(X_train, Y_train)
        
        predict_train = regr.predict(X_train)
        predict_test = regr.predict(X_test)
        
        df_train['predict'] = predict_train
        df_train['MSE'] = (np.array(df_train[cols_y]) - predict_train)**2
        
        df_test['predict'] = predict_test
        df_test['MSE'] = (np.array(df_test[cols_y]) - predict_test)**2
        
        mse_train = metrics.mean_squared_error(Y_train, predict_train)
        mae_train = metrics.mean_absolute_error(Y_train, predict_train)
        r2_train = metrics.r2_score(Y_train, predict_train)
        
        mse_test = metrics.mean_squared_error(Y_test, predict_test)
        mae_test = metrics.mean_absolute_error(Y_test, predict_test)
        
        stat_train = {"mse": mse_train, "mae": mae_train, "r2": r2_train}
        stat_test = {"mse": mse_test, "mae": mae_test}
        
        if printstat==True:
            print(f'train stat: {stat_train}')
            print(f'test stat: {stat_test}')
            # print(f'MSE Score manually calulated: {np.mean((np.array(df_final_vf[cols_y]) - predict)**2)}')

        
        return df_train, df_test, stat_train, stat_test

    def skpredict_window(self, skmodel, cols_x, cols_y, col_period , train_window=4 , test_window=1, test_gap = 0, expanding=False, print_iter=False):
        regr = skmodel
        train_test = self.train(self.df, col_period, train_window , test_window, test_gap, expanding)
        
        data_train = []
        data_test = []
        stat_train_times = []
        stat_test_times = []
        
        for i , j in enumerate(train_test):

            df_train = j[0]
            df_test = j[1]
            
            df_train, df_test, stat_train, stat_test = self.skpredict(df_train, df_test, regr, cols_x, cols_y, printstat=False)
            
            stat_train.update({"window": i, "date": df_train[col_period].unique().tolist()[0] + "-"+ df_train[col_period].unique().tolist()[-1]})
            stat_test.update({"window": i, "date": df_test[col_period].unique().tolist()[0] + "-"+ df_test[col_period].unique().tolist()[-1]})
            
            df_train['iteration'] = i
            df_test['iteration'] = i
        
            data_train.append(df_train)
            data_test.append(df_test)
            
            stat_train_times.append(stat_train)
            stat_test_times.append(stat_test)
            
            if print_iter != False:
                print(f'train stat: {stat_train}')
                print(f'test stat: {stat_test}\n')
            
        df_train_conso = pd.concat(data_train)
        df_test_conso = pd.concat(data_test)
        
        mse_train_all = df_train_conso.MSE.mean()
        mse_test_all = df_test_conso.MSE.mean()
        
        print(f'Average MSE train: {mse_train_all}')
        print(f'Average MSE test: {mse_test_all}')
        
        return df_train_conso , df_test_conso, stat_train_times, stat_test_times