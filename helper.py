from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, make_scorer, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def impute_mode(df, variable):
    '''
    Usage: replace NaN with the mode in specific column
    Input arguments:
    df  -- a dataframe object
    variable -- a column where you want to apply imputation
    
    Return: None
    '''
    # find most frequent category
    most_frequent_category = df.groupby([variable])[variable].count().sort_values(ascending=False).index[0]
    
    # replace NA
    df[variable].fillna(most_frequent_category, inplace=True)


def day_diff(df):
    '''
    Usage: calculate the day difference using columns "host_since" and "last_scraped"
    Input arguments:
    df  -- a dataframe object
    
    Return: None
    '''
    if ('last_scraped' in df.columns) & ('host_since' in df.columns):
        df['host_days'] = (pd.to_datetime(df['last_scraped']) - pd.to_datetime(df['host_since'])).apply(lambda x: x.days)
    else:
        print('Error: Date column does not exist in the dataset!')
        
def date2day(x):
    '''
    Usage: return day of the week based on the date provide
    Input arguments:
    x -- a date
    Return: a integer from 1-7 corresponding to Monday through Sunday
    '''
    if isinstance(x, datetime):
        return x.isoweekday()
    else:
        return x
    
def parse_dollar(x):
    '''
    Usage: parse a string object $xx.xx to a float number
    Input arguments:
    x -- an input string object
    Return: a float number corresponding to the exact amount of dollar
    '''
    if x is np.nan:
        return 0
    elif isinstance(x, str):
        return float(x[1:-3].replace(",", ""))
    else:
        return x
    
def parse_percentage(x):
    '''
    Usage: parse a string object xx% to a float number xx
    Input arguments:
    x -- an input string object
    
    Return: a float number corresponding to the percentage
    '''
    if isinstance(x, str):
        return float(x[0:-1])
    else:
        return x

def engineer_host_response_time(x):
    '''
    Usage: convert host_response_time to ordinal number 
    Input arguments:
    x -- an input string object
    
    Return: an ordinal number corresponding to how fast the host response is.
    '''
    if x == 'within an hour':
        return 1
    elif x == 'within a few hours':
        return 2
    elif x == 'within a day':
        return 3
    elif x == 'a few days or more':
        return 4
    else:
        return x

def engineer_cancellation_policy(x):
    '''
    Usage: convert cancellation_policy to ordinal number 
    Input arguments:
    x -- an input string object
    
    Return: an ordinal number corresponding to how strict the cancellation policy is.
    '''
    if x == 'flexible':
        return 1
    elif x == 'moderate':
        return 2
    elif x == 'strict':
        return 3
    elif x == 'super_strict_30':
        return 4
    else:
        return x

def encode_binary(x):
    '''
    Usage: convert string 't' and 'f' to 1 and 0, respectively
    Input arguments:
    x -- an input string object
    
    Return: an integer corresponding to true or false.
    '''
    if x == 'f':
        return 0
    elif x == 't':
        return 1
    else:
        return x

def inverse_transform(scaler, X, y):
    '''
    Usage: apply inverse transformation
    Input: 
    scaler - the scaler object
    X - the feature dataframe
    y - the label dataframe
    Return: inverse transformed feature and label dataframe
    '''
    data = pd.concat([X, y], axis = 1)
    data_inv = pd.DataFrame(data = scaler.inverse_transform(data), index = data.index, columns = data.columns)
    ncol = data_inv.shape[1]
    last_col = data_inv.columns[ncol-1]
    X_inv = data_inv.drop(labels = [last_col], axis = 1)
    y_inv = data_inv[[last_col]]
    return X_inv, y_inv
    
def GridSearch(X_train, X_test, y_train, y_test, criterion = ['mse'], n_estimators = [300, 600],
                  method = 'GBDT', learning_rate = 0.5, validate = False, cv = 5,
                  max_features = ['auto'], max_depth = [10, 20, 40], min_samples_leaf = [2,4], n_jobs = -1):
    '''
    Usage: use gridsearch to find optimal parameters for the random forest (RF) regressor.
    Input: training and testing sets from X and y variables
    Output: the best regressor
    '''
    
    if method == 'GB':
        clf = GradientBoostingRegressor(random_state=42, learning_rate = learning_rate)
    elif method == 'RF':
        clf = RandomForestRegressor(random_state=42, n_jobs = n_jobs)
    else:
        clf = RandomForestRegressor(random_state=42, n_jobs = n_jobs)
        
    parameters = {'criterion': criterion,
                  'n_estimators': n_estimators,
                  'max_depth': max_depth,
                  'min_samples_leaf':min_samples_leaf,
                  'max_features':max_features
                 }
    
    #Use gridsearch to find the best-model parameters.
    grid_obj = GridSearchCV(clf, parameters, cv = cv)
    grid_fit = grid_obj.fit(X_train, y_train)
    
    #obtaining best model, fit it to training set
    best_clf = grid_fit.best_estimator_
    best_clf.fit(X_train, y_train)

    # Make predictions using the new model.
    best_train_predictions = best_clf.predict(X_train)
    print('The training MSE Score is', mean_squared_error(y_train, best_train_predictions))
    print('The training R2 Score is', r2_score(y_train, best_train_predictions))
    
    if validate:
        best_test_predictions = best_clf.predict(X_test)
        print('The testing MSE Score is', mean_squared_error(y_test, best_test_predictions))
        print('The testing R2 Score is', r2_score(y_test, best_test_predictions))
    return best_clf

# distribution plots
def hist_plot(data, features):
    '''
    Usage: Plot distribution of missing/non-missing data with respect to the features
    Input:
    data - input dataframe
    features - selected features to plot
    Output: None
    '''
    nrow = 2
    ncol = int(len(features)/2)
    fig = plt.figure(figsize = (15, 5))
    for i in np.arange(len(features)):
        ax = fig.add_subplot(ncol, nrow, i+1)
        sns.distplot(data[data.has_NaN == True][features[i]], kde_kws = {'label': 'With NaN'})
        sns.distplot(data[data.has_NaN == False][features[i]], kde_kws = {'label': 'Without NaN'})
    return

def plot_timeseries_price(df):
    '''
    Usage: Plot 2 x 2 figures of time series data
    Input:
    data - input dataframe
    Output: the figure axis, processed data frames
    '''
    nrow = 2
    ncol = 2
    fig = plt.figure(figsize = (15, 10))
    # price vs time
    ax = fig.add_subplot(ncol, nrow, 1)
    data_ = df[['date', 'price', 'city']].groupby(['date', 'city']).mean().rename(columns = {"price":"avg_price"})
    data_.reset_index(inplace = True)
    sns.lineplot(x = 'date', y="avg_price", hue = 'city', data=data_, ax = ax)  
    # price vs time (zoom-in)
    ax = fig.add_subplot(ncol, nrow, 3)
    sns.lineplot(x = 'date', y="avg_price", hue = 'city', data=data_.iloc[87:120,:], ax = ax) 
    # price vs day
    ax = fig.add_subplot(ncol, nrow, 2)
    data = df[['day', 'price', 'city']].groupby(['day', 'city']).mean().reset_index().rename(columns = {"price":"avg_price"})
    #data.set_index('day').join(data[['day', 'price', 'city']].groupby(['city']).mean().set_index('day'))
    sns.barplot(x="day", y="avg_price", hue = 'city', data=data, ax = ax)
    ax.set_ylim(85, 105)
    
    # price difference vs day
    ax = fig.add_subplot(ncol, nrow, 4)
    min_price = data.groupby(['city']).min().drop(labels = 'day', axis = 1).rename(columns = {"avg_price":"min_price"})
    data = data.join(min_price, on = 'city')
    data['price_diff'] = data['avg_price'] - data['min_price']
    sns.barplot(x="day", y="price_diff", hue = 'city', data=data, ax = ax)
    # calculate normalized factor for price-day variation
    data1 = data[data['city'] == 'Boston']
    data2 = data[data['city'] == 'Seattle']
    
    fac1 = data1['avg_price']/data1['avg_price'].min()
    fac2 = data2['avg_price']/data2['avg_price'].min()
    fac3 = np.multiply(fac1,fac2)/np.min(np.multiply(fac1,fac2)).tolist()
    
    df1 = pd.DataFrame(data = fac1).reset_index(drop=True).rename(columns = {"avg_price":"day_factor"})
    df1['day'] = pd.DataFrame(data = 1 + np.arange(7))
    df1.set_index('day', inplace = True)
    
    df2 = pd.DataFrame(data = fac2).reset_index(drop=True).rename(columns = {"avg_price":"day_factor"})
    df2['day'] = pd.DataFrame(data = 1 + np.arange(7))
    df2.set_index('day', inplace = True)
    
    df3 = pd.DataFrame(data = fac3).reset_index(drop=True).rename(columns = {"avg_price":"day_factor"})
    df3['day'] = pd.DataFrame(data = 1 + np.arange(7))
    df3.set_index('day', inplace = True)
    data = data.set_index('day').join(df3, on='day')
    return ax, [data_, df1, df2, df3]

def plot_dist(df, x = 'bedrooms', y = 'price', split = False, fig = None, nrows = 1, ncols = 1, figsize = (10, 5)):
    '''
    Usage: plot the distribution
    Input:
    df - input dataframe
    x - plotted variable name on x-axis
    y - plotted variable name on y-axis
    split - True: show Boston and Seattle data in separate plots
            False: show Boston and Seattle data in one plot
    plot_dist - True: a one plot to show the distribution of x-components, only applied when split is False
    figsize - size of the figure  
    Output: axis of the plot
    '''
    plot_data = df.groupby(by = ['city', x]).mean()[[y]].reset_index()
    plot_data = plot_data.sort_values(by = y, ascending = True)
    bos_data = plot_data[plot_data.city == 'Boston'].sort_values(by = y, ascending = True)
    sea_data = plot_data[plot_data.city == 'Seattle'].sort_values(by = y, ascending = True)

    hue = 'city'
    if split:
        ax = bos_data.plot.bar(x=x, y=y, figsize = figsize, title = 'Boston')
        ax.set_ylim(50, 350)
        ax = sea_data.plot.bar(x=x, y=y, figsize = figsize, title = 'Seattle')
        ax.set_ylim(50, 350)
    else:
        if fig is None:
            fig = plt.figure(figsize = figsize)
            ax = fig.add_subplot()            
        else:
            ax = fig.add_subplot(nrows, ncols, 1)
        ax = sns.barplot(data = plot_data, x = x, y = y, hue = hue, ax = ax)
        ax.set_title('Averaged Price based on '+ x)

    print('Boston Price Ranged From {} to {}, average: {}' \
          .format(int(bos_data.price.min()), int(bos_data.price.max()), int(bos_data.price.mean())))
    print('Seattle Price Ranged From {} to {}, average: {}' \
          .format(int(sea_data.price.min()), int(sea_data.price.max()), int(sea_data.price.mean())))
    return ax

# count plots
def count_plot(data, feature, hue, ax = None, normalized = False):
    '''
    Usage: plot the distribution
    Input:
    data - input dataframe
    feature, hue - names of variables in data
    ax - size of the figure  
    normalized - True: show in percentage
               - False: show in count
    Output: axis of the plot
    '''
    if ax is None:
        fig = plt.figure(figsize = (20, 20))
    else:
        fig = ax.get_figure()
    
    ax = fig.add_subplot()
    if normalized is False:
        ax = sns.countplot(x = feature, hue = hue, data = data, ax = ax)
    else:
        data_ = (data.groupby([hue])[feature].value_counts(normalize=True).rename('percentage').mul(100)\
                 .reset_index().sort_values(feature))
        ax = sns.barplot(x=feature, y="percentage", hue=hue, data=data_, ax = ax)
        ax.set_title('Percent Count Distribution')
    return ax

def process_amenity(df):
    '''
    Usage: process amenity column in the dataframe
    Input: 
    df - the dataframe containing amenities column
    Return: a dictionary containing all unique amenity, 
            with each key linked to a list of dataframe index of row
            containing that amenity
    '''
    if 'amenities' not in df.columns:
        return
    
    listings = df['amenities'].tolist()
    listing_amenity = dict()
    for i, listing in enumerate(listings):
        amenities = listing.split(',')
        for amenity in amenities:
            amenity = re.sub(r'[^a-zA-Z]', "", amenity)
            listing_amenity.setdefault(amenity, list())
            listing_amenity[amenity].append(i)
    return listing_amenity

def add_new_amenity(df, listing_amenity, new_feature):
    '''
    Usage: add categorical variable of one particular amenity to the dataframe
    Input: 
    df - the dataframe to which you want to add the amenity category variable
    listing_amenity - a dictionary of amenities which contains lists of indexes
    new_feature - the name of amenity category variable 
    Return: None
    '''
    arr = np.zeros(df.shape[0], dtype=int)
    items = listing_amenity[new_feature]
    for item in items:
        arr[item] = 1
    df[new_feature] = arr
