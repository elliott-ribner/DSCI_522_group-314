# author: Elliott Ribner
# date: 2020-01-20

'''This script take a url and downloads a csv in the data directory. You should input url that is of .xls type.
Has default url value of default credit dataset if you do not send the url param as a command line argument. 
Override the default by sending in a url like exemplified below.

Usage: src/dl_xls_to_csv.py --output=<output> [--url=<url>] 

Options:
--output=<output>  Name of file to be saved, to be stored in /data directory. Include .csv filetype.
[--url=<url>]  Url of data to download, must be of xsl type.
'''

import pandas as pd
from docopt import docopt

opt = docopt(__doc__)


def main(output, url="https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"):
    # download xls to pandas dataframe
    df = pd.read_excel(url, encoding="utf-8")
    # save file as .csv type in the data directory
    df.to_csv(r"./data/%s" % (output))


if __name__ == "__main__":
    if (opt["--url"]):
        main(url=opt["--url"], output=opt["--output"])
    else:
        main(output=opt["--output"])
