# author: Zoe Pan
# date: 2020-01-21

'''This script do exploratory data analysis on cleaned data set and produce
   summary table, attributes distribution plots and variables relationship boxplots and bar plots.

Usage: src/eda_plots.py --filepath=<filepath> --outdir=<ourdir>  [<webbrowser>]

Options:
--filepath=<filepath>  Path of cleaned .csv file to download
--outdir=<outdir>  Name of directory to be saved in, no slashes nesscesary, 'results' folder recommended.
[<webbrowser>]  Optional, add 'firefox' to the argument if your web browser is Mozilla Firefox  
                Otherwise, default web browser is Google chrome  

Examples: python src/eda_plots.py --filepath=data/cleaned-credit-default-data.csv --outdir=results firefox
'''

import pandas as pd
from sklearn.model_selection import train_test_split
import altair as alt
import selenium
import seaborn as sns
from docopt import docopt
opt = docopt(__doc__)


def main(filepath, outdir, webbrowser='chrome'):

    #read data
    df = pd.read_csv(f"./{filepath}", index_col=0)

    #create lists for numeric and categorical features
    numeric_features = ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2',
       'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
       'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

    categorical_features = ['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_1', 'PAY_2',
           'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

    # split training and test
    X = df.drop(columns=['DEFAULT_NEXT_MONTH'])
    y = df['DEFAULT_NEXT_MONTH']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=122)
    df_train = X_train.copy()
    df_train['DEFAULT_NEXT_MONTH'] = y_train
    df_train_melt = df_train.melt(id_vars=['DEFAULT_NEXT_MONTH'],
                             value_vars=numeric_features,
                             var_name='Numeric features',
                             value_name='value')

    #save overview & summary table as .csv
    #df_train.head().to_csv(f'./{outdir}/head.csv')
    X_train[numeric_features].describe().to_csv(f'./{outdir}/num_describe.csv')
        
    #Numeric features correlations
    corr = X_train[numeric_features].corr()
    corr_df = corr.reset_index().melt(id_vars='index', var_name='yaxis', value_name='correlation')
    num_corr_chart = alt.Chart(corr_df).mark_rect().encode(
        alt.X('index:N', axis=alt.Axis(title=None)),
        alt.Y('yaxis:N', axis=alt.Axis(title=None)),
        alt.Color('correlation:Q')
    ).properties(title='Correlations between numeric features')
    num_corr_chart.save(f'./{outdir}/num_corr_chart.png', scale_factor=2, webdriver=webbrowser)

    #Numeric and response variable correlations
    num_res_chart = voilinplot(df_train_melt, 'Numeric features');
    num_res_chart.savefig(f'./{outdir}/num_res_chart.png')
    
    #Categorical and response variable correlations
    ncol = 2  #how many to display in each row
    row = alt.vconcat()
    col = alt.hconcat()
    for i in range(len(categorical_features)):
        col |= stack_bar(df_train, categorical_features[i], 'DEFAULT_NEXT_MONTH')
        if (i+1)%ncol==0:
            row &= col
            col = alt.hconcat()  
    cat_res_chart = row & col  
    cat_res_chart.save(f'./{outdir}/cat_res_chart.png', scale_factor=2, webdriver=webbrowser)


def num_bar(data, fea):
    """
    Parameters:
    --------------
    data: dataframe that has numeric column
    fea: string, name of the numeric column

    Returns:
    -------------
    altair histogram chart for numeric features

    """
    return alt.Chart(data).mark_bar().encode(
        alt.X(fea, type='quantitative', bin=alt.Bin(maxbins=10)),
        alt.Y('count()')
    ).properties(title=f'Count of {fea}', height=150, width=200)

def cat_bar(data, fea):
    """
    Parameters:
    --------------
    data: dataframe that has categorical column
    fea: string, name of the categorical column

    Returns:
    -------------
    altair bar chart for count of categorical features

    """
    return alt.Chart(data).mark_bar().encode(
        alt.X(fea, type='nominal'),
        alt.Y('count()')
    ).properties(title=f'Count of {fea}', height=200)
    
def voilinplot(data, cat):
    """
    Parameters:
    --------------
    data: dataframe that has categorical and numeric columns
    cat: string, name of the column that stores all categorical features name

    Returns:
    -------------
    seaborn voilin plot for numeric feature's distributions 
    at different categorical class values - faceted for all features
    """
    g = sns.FacetGrid(data, col=cat, col_wrap=4, height=5, sharey=False)
    g.map(sns.violinplot, 'DEFAULT_NEXT_MONTH', 'value');
    return g
    
def stack_bar(data, cat_var, res_var_cat):
    """
    Parameters:
    --------------
    data: dataframe that has categorical and response variable columns
    cat_var: string, name of the categorical column
    res_var_cat: string, name of the response variable column(categorical variable)

    Returns:
    -------------
    altair stackbar chart for categorical feature's distributions
    at different response variable's class values
    """
    return alt.Chart(data).mark_bar().encode(
    alt.Y(res_var_cat, type='nominal', title='Default'),
    alt.X('count()', axis=alt.Axis(grid=False), stack='normalize', title=f'Count of {cat_var} - Normalized'),
    alt.Color(cat_var, type='nominal')
).properties(width=400)

def test(filepath):
    """
    filepath: path of the .csv file
    Confirm correct dataset is read in for exploratory data analysis
    """
    df = pd.read_csv(f"./{filepath}", index_col=0)
    assert df.shape == (30000, 24)
    assert set(df.columns) == set(['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_1', 'PAY_2',
       'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
       'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
       'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',
       'DEFAULT_NEXT_MONTH']), 'Column names don\'t match, wrong path of the data provided'


if __name__ == "__main__":
    if opt['<webbrowser>']:
        test(opt['--filepath'])
        main(opt['--filepath'], opt['--outdir'], opt['<webbrowser>'])
    else:
        test(opt['--filepath'])
        main(opt['--filepath'], opt['--outdir'])
