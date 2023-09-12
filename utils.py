import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

def preprocess_csb(df, n_var, yr):
    """
    Pre-process the CSB data for modeling
    df: raw csb dataframe
    n_var: number of variables to be used for the modeling, e.g. 3
    yr: prediction year, e.g. 22 for 2022
    """
    if yr < 23:
        col_y = 'R' + str(yr)
        df_y = df[col_y]
        #subset columns from yr-n_var to yr-1
        col_x = ['R' + str(i) for i in range(yr-n_var, yr)]
        df_x = df[col_x]
        #Rename the columns in col_x: call it yr_1, yr_2, yr_3, etc. 
        df_x.columns = ['yr' + str(i) for i in range(1, n_var+1)]
        #Encode the categorical variables
        df_x = pd.get_dummies(df_x)
        df_y = LabelEncoder().fit_transform(df_y)
        #Add the CSBACRES column to df_x
        df_x['CSBACRES'] = df['CSBACRES']
        #Add the ASD column to df_x
        df_stateasd = pd.get_dummies(df['STATEASD'])
        #rename the columns in df_stateasd
        df_stateasd.columns = ['STATEASD_' + str(i) for i in df_stateasd.columns]
        df_x = pd.concat([df_x, df_stateasd], axis=1)

        #Return the processed data
        return df_x, df_y
    else: 
        col_x = ['R' + str(i) for i in range(yr-n_var, yr)]
        df_x = df[col_x]
        #Rename the columns in col_x: call it yr_1, yr_2, yr_3, etc. 
        df_x.columns = ['yr' + str(i) for i in range(1, n_var+1)]
        #Encode the categorical variables
        df_x = pd.get_dummies(df_x)
        #Add the CSBACRES column to df_x
        df_x['CSBACRES'] = df['CSBACRES']
        #Add the ASD column to df_x
        df_stateasd = pd.get_dummies(df['STATEASD'])
        #rename the columns in df_stateasd
        df_stateasd.columns = ['STATEASD_' + str(i) for i in df_stateasd.columns]
        df_x = pd.concat([df_x, df_stateasd], axis=1)
        #return the processed data
        return df_x
    

# Example usage:
# best_rf_model = hyperparameter_tuning(X, y, model_type='rf', n_iter=100, cv=5, random_state=42)


# Formulas to Calculate VIs 
def calc_NDWI(df):
    Green = df['B3']
    SWIR = df['B11']
    return (Green - SWIR) / (Green + SWIR)

def calc_NDVI(df):
    Red = df['B4']
    NIR = df['B8']
    return (NIR - Red) / (NIR + Red)

def calc_LSWI(df):
    NIR = df['B8']
    SWIR = df['B11']
    return (NIR - SWIR) / (NIR + SWIR)

def calc_NDRE(df):
    RedEdge = df['B5']
    NIR = df['B8']
    return (NIR - RedEdge) / (NIR + RedEdge)

def calc_Clgreen(df):
    NIR = df['B8']
    Green = df['B3']
    return NIR / Green-1

def calc_Clrededge(df):
    NIR = df['B8']
    RedEdge = df['B5']
    return NIR / RedEdge-1

def calc_Datt99(df):
    NIR = df['B8']
    RedEdge = df['B5']
    Red = df['B4']
    return (NIR - RedEdge) / (NIR - Red)

def calc_REP(df):
    Rededge1 = df['B5']
    Rededge2 = df['B6']
    Rededge3 = df['B7']
    Red = df['B4']
    return 705+(35*(0.5*(Red + Rededge3) - Rededge1)/(Rededge2 - Rededge1))

def calc_GWCCI(df):
    NIR = df['B8']
    Red = df['B4']
    SWIR = df['B11']
    NDVI = (NIR - Red) / (NIR + Red)
    return NDVI*SWIR

## Apply the VIs to the dataframe
def apply_VIs(df):
    df['NDWI'] = calc_NDWI(df)
    df['NDVI'] = calc_NDVI(df)
    df['LSWI'] = calc_LSWI(df)
    df['NDRE'] = calc_NDRE(df)
    df['Clgreen'] = calc_Clgreen(df)
    df['Clrededge'] = calc_Clrededge(df)
    df['Datt99'] = calc_Datt99(df)
    df['REP'] = calc_REP(df)
    df['GWCCI'] = calc_GWCCI(df)
    return df

# Clean up the GEE output 
def clean_gee(df, sdate = "03-01", edate = "05-31"):
    # df = pd.read_csv(path_csv)
    # system:index column convert to date using the first 10 characters of the index
    df['Date'] = df['system:index'].str[:8]
    # convert to datetime
    df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
    # Drop system:index column
    df = df.drop(columns=['system:index'])
    # Drop .geo column
    df = df.drop(columns=['.geo'])
    #change colunm type of CSBACRES and CSBID
    df['CSBACRES'] = df['CSBACRES'].astype('float64')
    df['CSBID'] = df['CSBID'].astype('str')
    df['Year'] = df['Date'].dt.year

    # Based on the Year column, convert sdate and edate to datetime
    sdate = pd.to_datetime(sdate + '-' + df['Year'].astype(str)[0])
    edate = pd.to_datetime(edate + '-' + df['Year'].astype(str)[0])
    #10 days before sdate, 10 days after edate
    sdate_10 = sdate - pd.Timedelta(days=10)
    edate_10 = edate + pd.Timedelta(days=10)
    # Subset the data based on sdate and edate
    df = df[(df['Date'] >= sdate_10) & (df['Date'] <= edate_10)]
    

    # Calculate VIs
    ## Filter the Bands if there are any values that are less than 0 or greater than 1
    Bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B11']
    for band in Bands:
        df = df[(df[band] >= 0) & (df[band] <= 1)]
    ## Apply the VIs to the dataframe
    df = apply_VIs(df)
    VIs = ['NDWI', 'NDVI', 'LSWI', 'NDRE', 'Clgreen', 'Clrededge', 'Datt99', 'REP', 'GWCCI']
    
    # Resample to every 10 days
    date_range = pd.date_range(start = sdate, end = edate, freq = '10D')

    # Set 'Dates' column as the index
    df.set_index(['CSBID', 'Date'], inplace=True)

    # Aggregate any duplicates
    df = df.groupby(['CSBID', 'Date']).mean()

    # Define a date range from March 01 to June 01
    date_range = pd.date_range(start=sdate_10, end=edate_10)

    # Function to reindex and then resample
    def process_group(group):
        group = group.reset_index(level=0, drop=True)  # Drop CSBID from index for this operation
        group = group.reindex(date_range)#.fillna(method='ffill').fillna(method='bfill')
        return group.resample('10D').mean()

    df_resample = df[VIs].groupby('CSBID').apply(process_group)
    df_resample.reset_index(level=1, inplace=True)
    df_resample.index.name = 'CSBID'
    df_resample.reset_index(inplace=True)
    df_resample.rename(columns={'level_1': 'Date'}, inplace=True)

    # Interpolate the missing values for each CSBID
    df_resample = df_resample.sort_values(by = ['CSBID', 'Date'])
    def interp_group(group):
        group[VIs] = group[VIs].interpolate(method = 'linear', limit_direction = 'both')
        return group
    df_interp = df_resample.groupby('CSBID').apply(interp_group)

    # Calculate DOY and Year
    df_interp['DOY'] = df_interp['Date'].dt.dayofyear
    df_interp['Year'] = df_interp['Date'].dt.year

    #Subset the data based on sdate and edate
    df_interp = df_interp[(df_interp['Date'] >= sdate) & (df_interp['Date'] <= edate)]
    df_resample = df_resample[(df_resample['Date'] >= sdate) & (df_resample['Date'] <= edate)]

    return(df_resample,df_interp)



# Convert long to wide and output 
def long_to_wide(df_long, Output = False, path_out = None):
    # Time series of VIs
    VIs = ['NDWI', 'NDVI', 'LSWI', 'NDRE', 'Clgreen', 'Clrededge', 'Datt99', 'REP', 'GWCCI']
    #convert long to wide
    df_wide = df_long.pivot(index='CSBID', columns='DOY', values=VIs)
    # Rename the df_wide columns 
    df_wide.columns = ['_'.join(str(s).strip() for s in col if s) for col in df_wide.columns]
    df_wide = df_wide.reset_index()
    # Output 
    if Output == True:
        df_wide.to_csv(path_out, index=False)
    return(df_wide)


# ML model training
# ## Preseason model training 
# def ml_model_preseason(csbdf, model, yr, X, Y, Xnew = None, Ynew = None):
#     """
#     csbdf: raw csb dataframe (Only used for record the results)
#     model: model name, e.g. 'rf', 'lgb', 'xgb'
#     yr: prediction year, e.g. 22 for 2022
#     X: X_train
#     Y: Y_train
#     Xnew: X_test
#     Ynew: Y_test
#     """
#     # #Generate dataframe 
    
#     #Train the model
#     clf_rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)
#     clf_rf.fit(X, Y)
#     Ypred_new = clf_rf.predict(Xnew)
#     if Ynew is not None:
#         Accuracy = accuracy_score(Ynew, Ypred_new)
#         CM = confusion_matrix(Ynew, Ypred_new)
#         print('Accuracy: ', Accuracy)
#         # print('Confusion Matrix: ', CM)
    
#     #Calculate total acres of each class
#     df_results = csbdf.copy()
#     df_results['R{}_PED'.format(yr)] = Ypred_new
#     #Predicted total acres
#     df_corn1 = df_results[df_results['R{}_PED'.format(yr)] == 0].groupby(['CNTYFIPS', 'CNTY']).agg({'CSBACRES': 'sum'}).reset_index() #Corn
#     df_corn1.rename(columns={'CSBACRES': 'CornAcres_Pred'}, inplace=True)
#     df_soy1 = df_results[df_results['R{}_PED'.format(yr)] == 2].groupby(['CNTYFIPS', 'CNTY']).agg({'CSBACRES': 'sum'}).reset_index() #Soybean
#     df_soy1.rename(columns={'CSBACRES': 'SoyAcres_Pred'}, inplace=True)
#     if Ynew is not None:
#         #Actual total acres
#         df_corn2 = df_results[df_results['R{}'.format(yr)] == 'Corn'].groupby(['CNTYFIPS', 'CNTY']).agg({'CSBACRES': 'sum'}).reset_index() #Corn
#         df_corn2.rename(columns={'CSBACRES': 'CornAcres_Actual'}, inplace=True)
#         df_soy2 = df_results[df_results['R{}'.format(yr)] == 'Soybean'].groupby(['CNTYFIPS', 'CNTY']).agg({'CSBACRES': 'sum'}).reset_index() #Soybean
#         df_soy2.rename(columns={'CSBACRES': 'SoyAcres_Actual'}, inplace=True)
#         #Merge the two dataframes
#         df_corn = pd.merge(df_corn1, df_corn2, how='inner', left_on=['CNTYFIPS','CNTY'], right_on=['CNTYFIPS','CNTY'])
#         df_soy = pd.merge(df_soy1, df_soy2, how='inner', left_on=['CNTYFIPS','CNTY'], right_on=['CNTYFIPS','CNTY'])
#         #Sum of the total acres
#         Sum_Corn_Pred = df_corn['CornAcres_Pred'].sum()
#         Sum_Corn_Actual = df_corn['CornAcres_Actual'].sum()
#         Sum_Soy_Pred = df_soy['SoyAcres_Pred'].sum()
#         Sum_Soy_Actual = df_soy['SoyAcres_Actual'].sum()
#         df_sum = pd.DataFrame({'YR':[yr],'Sum_Corn_Pred': [Sum_Corn_Pred], 'Sum_Corn_Actual': [Sum_Corn_Actual], 'Sum_Soy_Pred': [Sum_Soy_Pred], 'Sum_Soy_Actual': [Sum_Soy_Actual]})
#         df_sum['Diff_Corn'] = df_sum['Sum_Corn_Pred'] - df_sum['Sum_Corn_Actual']
#         df_sum['Diff_Soy'] = df_sum['Sum_Soy_Pred'] - df_sum['Sum_Soy_Actual']
#         df_sum['Diff_Corn_Percent'] = df_sum['Diff_Corn']/df_sum['Sum_Corn_Actual']*100
#         df_sum['Diff_Soy_Percent'] = df_sum['Diff_Soy']/df_sum['Sum_Soy_Actual']*100
#         print(df_sum)
        
#         return clf_rf, Ypred_new, Accuracy, CM, df_corn, df_soy, df_sum
#     else:
#         return clf_rf, Ypred_new, df_corn1, df_soy1


## With both preseason and in-season data
def ml_model_base(csbdf, model, yr, X, Y, test_size = 0.7, random_state = 777,Xnew=None, Ynew = None):
    if test_size == 0:
        X_train = X
        Y_train = Y
    else: 
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
    # Train the model 
    if model == 'rf':
        clf_model = RandomForestClassifier(random_state=random_state, n_estimators=100, max_depth=None, min_samples_split = 2)
        clf_model.fit(X_train, Y_train)
        # Predict on new years data 
        if test_size == 0:
            Ypred_new = clf_model.predict(Xnew)
        else:
            Ypred_test = clf_model.predict(X_test)
            Ypred_new = clf_model.predict(Xnew)
    elif model == 'lgb':
        params = {'objective': 'multiclass', 
                  'num_class': 3,
                  'metric': 'multi_logloss',
                  'num_leaves': 31,
                  'boosting_type': 'gbdt',
                  'learning_rate': 0.05,
                  'feature_fraction': 0.9,}
        # Train the model 
        num_round = 100
        clf_model = lgb.train(params, lgb.Dataset(X, Y), num_round, early_stopping_rounds=10)
        if test_size == 0:
            # New Years data
            Ypred_new = clf_model.predict(Xnew, num_iteration=clf_model.best_iteration)
            Ypred_new = [list(row).index(max(row)) for row in Ypred_new]
        else:
            # Current year test
            Ypred_test = clf_model.predict(X_test, num_iteration=clf_model.best_iteration)
            # Convert predictions to class labels
            Ypred_test = [list(row).index(max(row)) for row in Ypred_test]
            # New Years data
            Ypred_new = clf_model.predict(Xnew, num_iteration=clf_model.best_iteration)
            Ypred_new = [list(row).index(max(row)) for row in Ypred_new]
    
    else:
        print('Currently only support rf and lgb models')
    
    # if test_size == 0:
    #     return clf_model, Ypred_new
    # else:
    #     return clf_model, Ypred_test, Ypred_new
    # Calculate accuracy and confusion matrix
    
    if test_size != 0:
        Accuracy = accuracy_score(Y_test, Ypred_test)
        CM = confusion_matrix(Y_test, Ypred_test)
        print('Accuracy test set: ', Accuracy)
        print('Confusion Matrix test set: ', CM)
    if Ynew is not None:
        Accuracy_new = accuracy_score(Ynew, Ypred_new)
        CM_new = confusion_matrix(Ynew, Ypred_new)
        print('Accuracy new set: ', Accuracy_new)
        print('Confusion Matrix new set: ', CM_new)
    # Calculate total acres of each class
    df_results = csbdf.copy()
    df_results['R{}_PED'.format(yr)] = Ypred_new
    #Predicted total acres
    df_corn1 = df_results[df_results['R{}_PED'.format(yr)] == 0].groupby(['CNTYFIPS', 'CNTY']).agg({'CSBACRES': 'sum'}).reset_index() #Corn
    df_corn1.rename(columns={'CSBACRES': 'CornAcres_Pred'}, inplace=True)
    df_soy1 = df_results[df_results['R{}_PED'.format(yr)] == 2].groupby(['CNTYFIPS', 'CNTY']).agg({'CSBACRES': 'sum'}).reset_index() #Soybean
    df_soy1.rename(columns={'CSBACRES': 'SoyAcres_Pred'}, inplace=True)
    if Ynew is not None:
        #Actual total acres
        df_corn2 = df_results[df_results['R{}'.format(yr)] == 'Corn'].groupby(['CNTYFIPS', 'CNTY']).agg({'CSBACRES': 'sum'}).reset_index() #Corn
        df_corn2.rename(columns={'CSBACRES': 'CornAcres_Actual'}, inplace=True)
        df_soy2 = df_results[df_results['R{}'.format(yr)] == 'Soybean'].groupby(['CNTYFIPS', 'CNTY']).agg({'CSBACRES': 'sum'}).reset_index() #Soybean
        df_soy2.rename(columns={'CSBACRES': 'SoyAcres_Actual'}, inplace=True)
        #Merge the two dataframes
        df_corn = pd.merge(df_corn1, df_corn2, how='inner', left_on=['CNTYFIPS','CNTY'], right_on=['CNTYFIPS','CNTY'])
        df_soy = pd.merge(df_soy1, df_soy2, how='inner', left_on=['CNTYFIPS','CNTY'], right_on=['CNTYFIPS','CNTY'])
        #Sum of the total acres
        Sum_Corn_Pred = df_corn['CornAcres_Pred'].sum()
        Sum_Corn_Actual = df_corn['CornAcres_Actual'].sum()
        Sum_Soy_Pred = df_soy['SoyAcres_Pred'].sum()
        Sum_Soy_Actual = df_soy['SoyAcres_Actual'].sum()
        df_sum = pd.DataFrame({'YR':[yr],'Sum_Corn_Pred': [Sum_Corn_Pred], 'Sum_Corn_Actual': [Sum_Corn_Actual], 'Sum_Soy_Pred': [Sum_Soy_Pred], 'Sum_Soy_Actual': [Sum_Soy_Actual]})
        df_sum['Diff_Corn'] = df_sum['Sum_Corn_Pred'] - df_sum['Sum_Corn_Actual']
        df_sum['Diff_Soy'] = df_sum['Sum_Soy_Pred'] - df_sum['Sum_Soy_Actual']
        df_sum['Diff_Corn_Percent'] = df_sum['Diff_Corn']/df_sum['Sum_Corn_Actual']*100
        df_sum['Diff_Soy_Percent'] = df_sum['Diff_Soy']/df_sum['Sum_Soy_Actual']*100
        print(df_sum)
        
        if test_size == 0:
            return clf_model, Ypred_new, Accuracy_new, CM_new, df_corn, df_soy, df_sum
        else:
            return clf_model, Ypred_test, Ypred_new, Accuracy, CM, Accuracy_new, CM_new, df_corn, df_soy, df_sum
    else:
        if test_size == 0:
            return clf_model, Ypred_new, df_corn1, df_soy1
        else:
            return clf_model, Ypred_test, Ypred_new, Accuracy, CM, Ypred_new, df_corn1, df_soy1
        


def hyperparameter_tuning(X, y, model_type='rf', n_iter=20, cv=5, subset_frac = 0.1, random_state=None):
    """
    Perform hyperparameter tuning using RandomizedSearchCV for RF, LightGBM, or XGBoost.
    
    Parameters:
    - X, y: Input data and target labels
    - model_type: 'rf' for Random Forest, 'lightgbm' for LightGBM, 'xgboost' for XGBoost
    - n_iter: Number of parameter settings sampled in RandomizedSearchCV
    - cv: Number of cross-validation folds
    - subset_frac: Fraction of data to train on (for faster hyperparameter tuning)  
    - random_state: Random seed for reproducibility

    Returns:
    - Best model with optimized parameters
    """

    # Train on a subset of the data
    if subset_frac < 1.0:
        subset_idx = np.random.choice(len(y), int(subset_frac * len(y)), replace=False)
        X = X.iloc[subset_idx]
        y = y[subset_idx]
    else: 
        pass

    if model_type == 'rf':
        model = RandomForestClassifier(random_state=random_state)
        param_dist = {
            'n_estimators': np.arange(50, 501, 50),
            'max_features': ['auto', 'sqrt', 'log2'],
            'max_depth': np.arange(5, 31, 5),
            'min_samples_split': np.arange(2, 11, 2),
            'min_samples_leaf': np.arange(1, 11, 2)
        }
        
    elif model_type == 'lightgbm':
        model = lgb.LGBMClassifier(random_state=random_state)
        param_dist = {
            'learning_rate': np.logspace(-3, 0, 4),
            'n_estimators': np.arange(50, 501, 50),
            'max_depth': np.arange(5, 31, 5),
            'num_leaves': np.arange(20, 151, 30),
            'min_child_samples': np.arange(5, 51, 10),
            'feature_fraction': np.linspace(0.6, 1.0, 5),
            'bagging_fraction': np.linspace(0.6, 1.0, 5),
            'bagging_freq': np.arange(1, 11, 2)
        }
        
    elif model_type == 'xgboost':
        model = xgb.XGBClassifier(random_state=random_state, use_label_encoder=False, eval_metric="logloss")
        param_dist = {
            'learning_rate': np.logspace(-3, 0, 4),
            'n_estimators': np.arange(50, 501, 50),
            'max_depth': np.arange(5, 31, 5),
            'subsample': np.linspace(0.6, 1.0, 5),
            'colsample_bytree': np.linspace(0.6, 1.0, 5),
            'gamma': np.linspace(0, 0.5, 11),
            'alpha': np.logspace(-3, 1, 5),
            'lambda': np.logspace(-3, 1, 5)
        }
        
    else:
        raise ValueError("model_type should be 'rf', 'lightgbm', or 'xgboost'")

    # Cross validation strategy
    stratified_kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    # RandomizedSearchCV
    random_search = RandomizedSearchCV(model, param_distributions=param_dist, 
                                       n_iter=n_iter, scoring='accuracy', 
                                       cv=stratified_kfold, verbose=1, 
                                       n_jobs=-1, random_state=random_state)
    
    random_search.fit(X, y)

    return random_search.best_estimator_