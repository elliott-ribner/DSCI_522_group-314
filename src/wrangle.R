# author: Elliott Ribner
# date: 2020-01-20


"This script saves handles the cleaning and preprocessing of the dataset imported in dl_xls_to_csv.py.

Usage: src/wrangle.R <input> <output>

Options:
<input>  Name of csv file to download, must be within the /data directory.
<output>  Name of file to be saved, to be stored in /data directory. Include .csv filetype. 
" -> doc

library(tidyverse)
library(testthat)
library(docopt)

opt <- docopt(doc)

main <- function(input, output){
    df <- read_csv(paste('./data/', input, sep=""), skip=1)
    df <- df %>% select(-'0')
    df <- df %>% rename(
        DEFAULT_NEXT_MONTH= 'default payment next month',
        PAY_1= 'PAY_0'
    )
    write_csv(df, paste('./data/', output, sep=""))
}

test_main <- function(output){
  test_that("test main func should create csv", {
    df <- read_csv(paste('./data/', output, sep=""), skip=1)
        expect_equal(nrow(df), 29999)
  })
}

main(opt$input, opt$output)

test_main(opt$output)