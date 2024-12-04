import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as axes
import seaborn as sns


#importing of climate data
print('importing data')
df_climate = pd.read_csv('static/sub_saharan_africa_climate_data_1991_2023_v3-2.csv')

#importing of production crop data using latin-1 encoding 
df_crop = pd.read_csv("static/Production_Crops_E_Africa.csv", encoding="latin-1")
# clean data

#ALL CLEANED DATA ARE STILL IN TYPE DATAFRAME!!!

# for df_climate, sum all respective climate data per year for each country and find average

#prep columns for cleaned dataframe
columns = df_climate.columns.tolist()
columns.remove('Country/Region')

#make new dataframe with columns 
df_climate_cleaned = pd.DataFrame(columns = columns)

for year in range(1991,2020):
    # extract relevent data based on year remove country data
    data_year = pd.DataFrame(df_climate[df_climate['Year'] == year]).drop('Country/Region' , axis = 1)
    
    #add new entry into cleaned dataset
    df_climate_cleaned.loc[len(df_climate_cleaned)] = data_year.mean()

# print(df_climate_cleaned.shape)

#for df_crop, remove area harvest and yield, combine all types of crops under the same country and change tonnes to mega tonnes
# print(df_crop)

non_SSA = ['Algeria', 'Libya', 'Tunisia', 'Egypt', 'Morocco']

df_crop_cleaned = pd.DataFrame()
for year in range(1991,2020):
    year = "Y" + str(year)
    for area in df_crop['Area'].unique():
        # print(year, area)
        #conditional to be sub-saharan
        if area not in non_SSA:
            #filter data for that year based on selected area and getting only 
            area_data = pd.DataFrame(df_crop[(df_crop['Area'] == area) & (df_crop['Element'] == 'Production')][[year]])
            # print(area, area_data)

            #Sum and convert to megatonnes
            total_production = area_data.sum()/ 1000000 
            # print(total_production, area)

            #add to cleaned dataframe
            df_crop_cleaned.loc[year,area] = total_production.iloc[0]
    total_year = df_crop_cleaned.loc[year,:].sum()
    # print(total_year)
    
    #find sum of production for the whole year
    df_crop_cleaned.loc[year, 'total'] = total_year
    df_crop_cleaned = df_crop_cleaned.reset_index(drop = True)
    
# print(df_crop_cleaned.shape)
#merge both data set into 1
df =  pd.concat([df_climate_cleaned, df_crop_cleaned], axis = 1)

# print(df)

#Relevant Functions

def get_features_targets(df: pd.DataFrame, 
                         feature_names: list[str], 
                         target_names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_feature = pd.DataFrame(df.loc[:,feature_names])
    df_target = pd.DataFrame(df.loc[:,target_names])
    return df_feature, df_target

def normalize_minmax(array_in: np.ndarray, columns_mins: np.ndarray=None, 
                     columns_maxs: np.ndarray=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    out = array_in.copy()
    if columns_mins is None:
        columns_mins = np.min(out, axis=0)
        columns_mins = columns_mins.reshape(1, -1)  # Reshape to 2D (1 row, n columns)
    
    if columns_maxs is None:
        columns_maxs = np.max(out, axis=0)
        columns_maxs = columns_maxs.reshape(1, -1)  
    out = (out-columns_mins)/(columns_maxs-columns_mins)
    return out, columns_mins, columns_maxs

def prepare_feature(np_feature: np.ndarray) -> np.ndarray:
    no_row = np_feature.shape[0]
    ones = np.ones((no_row,1))
    return np.concatenate((ones,np_feature), axis =1)

def split_data(df_feature: pd.DataFrame, df_target: pd.DataFrame, 
               random_state: int=None, 
               test_size: float=0.5) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
    np.random.seed(random_state)
    
    df_feature_test_index = np.random.choice(df_feature.index, size = int(test_size*df_feature.shape[0]), replace = False)
    
    df_feature_test = df_feature.loc[df_feature_test_index,:]
    df_feature_train = df_feature.drop(df_feature_test_index, axis = 0)
    
    df_target_test = pd.DataFrame(df_target.loc[df_feature_test_index,:])
    df_target_train = df_target.drop(df_feature_test_index, axis = 0)
    
    return df_feature_train, df_feature_test, df_target_train, df_target_test

def predict_linreg(array_feature: np.ndarray, beta: np.ndarray, mins, maxs) -> np.ndarray:
    # normalize array_feature
    normalized_array, mins, maxs = normalize_minmax(array_feature, mins , maxs)

    #add ls to array_feature matrix 
    array_feature = prepare_feature(normalized_array)

    #predict (use calc_linreg)
    prediction = calc_linreg(array_feature, beta)
    return prediction

def calc_linreg(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return np.matmul(X,beta)

def r2_score(y: np.ndarray, ypred: np.ndarray) -> float:
    ss_res = np.sum((y - ypred)**2)
    ss_tot = np.sum((y - np.mean(ypred))**2)
    return 1-(ss_res/ss_tot)

def adjusted_r2(r2, n, p):
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

def mean_squared_error(target: np.ndarray, pred: np.ndarray) -> float:
    n = target.shape[0]
    mse = (1/n)*(np.sum((target-pred)**2))
    return mse

def compute_cost_linreg(X: np.ndarray, y: np.ndarray, beta: np.ndarray) -> np.ndarray:
    y_hat = np.matmul(X,beta)
    J = (1/(2*(X.shape[0]))) * np.matmul((y_hat - y).T, (y_hat - y))
    return np.squeeze(J)

def gradient_descent_linreg(X: np.ndarray, y: np.ndarray, beta: np.ndarray, 
                            alpha: float, num_iters: int) -> tuple[np.ndarray, np.ndarray]:
    J_storage = []
    for i in range(num_iters):
        J_storage.append(compute_cost_linreg(X, y, beta))
        beta = beta - np.matmul((alpha/X.shape[0])*X.T , (calc_linreg(X,beta) - y))
    return beta, J_storage

def apply_power(array):
    polynormial_degree = [5, 6, 9, 6, 7, 7, 10, 5, 4, 9, 10, 6, 6]
    for idx in range(len(polynormial_degree)):
        power = polynormial_degree[idx]
        array[0,idx] = array[0,idx]**power
    return array

#apply polynormial power to the respective column data

#seperate into features data and targets
features_name = columns.copy() #col of features


df_p_features, df_p_targets = get_features_targets(df, features_name, 'total') #this is in type DataFrame 
polynormial_degree = [5, 6, 9, 6, 7, 7, 10, 5, 4, 9, 10, 6, 6]
for idx in range(len(polynormial_degree)):
    power = polynormial_degree[idx]
    df_p_features.iloc[:,idx] = df_p_features.iloc[:, idx]**power
    
    
#still in type dataframe, split the data into training and testing sets
df_p_features_train, df_p_features_test, df_p_targets_train, df_p_targets_test = split_data(df_p_features, df_p_targets, 0, 0.3)

#convert to all training dataframes to numpy
#normalize the training features based on min_max
np_p_features_train_norm, mins_p, maxs_p = normalize_minmax(df_p_features_train.to_numpy())

np_p_targets_train = df_p_targets_train.to_numpy()

#prep features
np_p_features_train_norm = prepare_feature(np_p_features_train_norm) 

iterations: int = 80000
alpha: float = 0.02
beta_p: np.ndarray = np.zeros((np_p_features_train_norm.shape[1],1))

#do gradient descent
print("training")
beta_p, J_storage = gradient_descent_linreg(np_p_features_train_norm, np_p_targets_train, beta_p, alpha, iterations)

# put Python code to test & evaluate the model
y_p_hat = predict_linreg(df_p_features_test.to_numpy(), beta_p, mins_p ,maxs_p)
# y_p_hat = np.abs(y_p_hat)
# print(y_p_hat)
# print(df_p_targets_test)
r2_p = r2_score(df_p_targets_test.to_numpy(), y_p_hat)
a_p_r2 = adjusted_r2(float(r2_p), df_p_features_train.shape[0], df_p_features_test.shape[1])

mse_p = mean_squared_error(df_p_targets_test.to_numpy(), y_p_hat)
rmse_p = mse_p**0.5

_, tmins_p, tmaxs_p = normalize_minmax(df_p_targets_test.to_numpy())
rmse_p_percentage = float((rmse_p/(tmaxs_p-tmins_p))*100)

print('ml.py completed')