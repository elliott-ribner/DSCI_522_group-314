# author: Elliott Ribner
# date: 2020-01-20

'''This script saves handles the cleaning and preprocessing of the dataset imported in dl_xls_to_csv.py.

Usage: src/wrangle.py --input=<input> --output=<output>

Options:
--input=<input>  Name of csv file to download, must be within the /data directory.
--output=<output>  Name of file to be saved, to be stored in /data directory. Include .csv filetype. 
'''

import pandas as pd
from docopt import docopt

opt = docopt(__doc__)


def main(input, output):
    # download xls to pandas dataframe
    df = pd.read_csv(f"./data/{input}", index_col=1, skiprows=1)
    df = df.drop(columns=['0'])
    df = df.rename(
        columns={'default payment next month': 'DEFAULT_NEXT_MONTH', 'PAY_0': 'PAY_1'})
    df.to_csv(r"./data/%s" % (output))


if __name__ == "__main__":
    main(input=opt["--input"], output=opt["--output"])
