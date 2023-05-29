import pandas as pd
import numpy as np
import scipy as sp
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def clean_data():
    sc_player_data = pd.read_csv('data/external/starcraft_player_data.csv')

    # Fixing dtypes and replacing ? with np.nan
    sc_player_data = sc_player_data.replace('?', np.nan)
    sc_player_data = sc_player_data.astype({'Age': 'float','HoursPerWeek':'float','TotalHours':'float'}) # since np nan is a float

    # Removing outliers
    sc_player_data = sc_player_data[sc_player_data['TotalHours'] != 1000000]
    sc_player_data = sc_player_data[sc_player_data['HoursPerWeek'] != 168] 

    sc_player_data.to_csv('data/interim/cleaned_sc_player_data.csv', index=False)

    return sc_player_data

def impute_na_data():
    cleaned_sc_player_data = pd.read_csv('data/interim/cleaned_sc_player_data.csv')

    # Replacing NA values in Age with mean
    cleaned_sc_player_data['Age'] = cleaned_sc_player_data['Age'].fillna(cleaned_sc_player_data['Age'].mean())
    # 40 hr week
    cleaned_sc_player_data['HoursPerWeek'] = cleaned_sc_player_data['HoursPerWeek'].fillna(40) 
    # Fitting a curve to interpolate TotalHours value
    mean_total_hours = cleaned_sc_player_data.groupby(by=['LeagueIndex'])['TotalHours'].mean().reset_index()
    
    config = {'TotalHours':interpolate}               

    interpolated = mean_total_hours.agg(config)
    interpolated['LeagueIndex'] = mean_total_hours['LeagueIndex']

    cleaned_sc_player_data['TotalHours'].fillna(interpolated['TotalHours'].iloc[7],inplace=True)

    cleaned_sc_player_data.to_csv('data/interim/imputed_sc_player_data.csv', index=False)

    return cleaned_sc_player_data

# Helper Function for impute_na_data
def f(x, m, c):
    return m*x**2+c 
# Helper Function for impute_na_data
def interpolate(s):
    temp = s.dropna()
    popt, pcov = sp.optimize.curve_fit(f, temp.index, temp)
    output = [i for i in f(s.index, *popt)] 
    return pd.Series(output)

def remove_correlated_data(threshold):
    imputed_sc_player_data = pd.read_csv('data/interim/imputed_sc_player_data.csv')

    identified_correlated_features = identify_correlated_features(imputed_sc_player_data,threshold)

    imputed_sc_player_data = imputed_sc_player_data.drop(identified_correlated_features,axis=1)

    imputed_sc_player_data.to_csv('data/interim/removed_corr_sc_player_data.csv',index=False)

    return imputed_sc_player_data

# Helper function for remove_correlated_data that identifies correlated features that are also the least correlated to target variable.
def identify_correlated_features(data, threshold):
    identified_features = set()  
    corr_matrix = data.corr()
    
    # LeagueIndex and Feature correlations
    rank_correlation = {}
    
    for i in range(len(corr_matrix.columns)):    
        for j in range(len(corr_matrix.columns)):
            
            if i == 0: # populate correlations to rank by going through first row only 
                rank_correlation[corr_matrix.columns[j]] = corr_matrix.iloc[i,j]
                           
            else:
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    column_name = corr_matrix.columns[i]  
                    row_name = corr_matrix.columns[j]
                    
                    if column_name != row_name: 
                        if rank_correlation[column_name] > rank_correlation[row_name]:
                            identified_features.add(row_name)
                        else:
                            identified_features.add(column_name)
    return identified_features

def balance_data(): # deal with class imbalance with oversampling (SMOTE)
    removed_corr_sc_player_data = pd.read_csv('data/interim/removed_corr_sc_player_data.csv')

    X = removed_corr_sc_player_data.drop('LeagueIndex',axis=1)  
    y = removed_corr_sc_player_data['LeagueIndex'] 

    oversample = SMOTE()
    X, y = oversample.fit_resample(X, y)

    X['LeagueIndex'] = y
    removed_corr_sc_player_data = X

    removed_corr_sc_player_data.to_csv('data/processed/processed_sc_player_data.csv',index=False)

    return removed_corr_sc_player_data

if __name__ == '__main__': 
    clean_data()
    impute_na_data()
    threshold = .7
    remove_correlated_data(threshold)
    balance_data()
