# author: Elliott Ribner
# date: 2020-01-16

'''This script take a url and downloads a csv in the data directory. You should input url that is of .xls type.
Has default url value of default credit dataset if you do not send the url param as a command line argument. 
Override the default by sending in a url like exemplified below.

Usage: src/dl_xls_to_csv.py [--url=<url>] 

Options:
[--url=<url>]  Url of data to download, must be of xsl type.
'''

import pandas as pd
import xlrd
from docopt import docopt

opt = docopt(__doc__)

def main(url="https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"):
    # download xls to pandas dataframe
    df = credit_default_data = pd.read_excel(url, encoding="utf-8")
    # save file as .csv type in the data directory
    df.to_csv(r'./data/cedit_default_data.csv')


print(opt["--url"])
if __name__ == "__main__":
    if (opt["--url"]):
        main(opt["--url"])
    else:
        main()
