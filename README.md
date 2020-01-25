# The Prediction of Customer Default  

Author: Elliott Ribner, Mohammed Salama, Zoe Pan  

This project was carried out as a part of the requirements MDS-UBC program  

## Introduction

The prediction of Customer default payments is an important issue in risk management by banks and developing accurate predictive tools is highly needed to mitigate losses associated with Credit Risk. In our project, we will use Logistic regression model and the related machine learning techniques to predict customer default payment and therefore help us answer our research question: what features strongly predict default payment.  

In this data analysis project, we will be using Data Set that is publicly available from [UCI Machine Learning Repository Irvine, CA: University of California, School of Information and Computer Science](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients) to find what features predict default payment. The data, while hosted by UCI, was originally publicized by Chung Hua University of Taiwan and Tamkang University of Taiwan. The data was collected in 2005 from the Taiwanese market. The data contain 30,000 instances in total and 23 features about customers' payment, bill histories and demographic factors.  

Due to imbalanced class in the response variable, we used [SMOTE](https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.SMOTE.html) (Synthetic Minority Oversampling Technique) to create a balanced data set to fit the model. [`RobustScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html) to scale predictors. And we used logistic regression model(`LogisticRegression`) and [`RFE`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html#sklearn.feature_selection.RFE)(recursive feature elimination) to prune and select features that predict default payment. We finally narrowed features from 23 down to 7 that predict default payment: Amount of the given credit (NT dollar), EDUCATION, MARRIAGE, AGE, Past monthly repayment status in September 2005, Past monthly repayment status in September 2005, Amount of previous payment (NT dollar) in September 2005. Comparing it with all features baseline model, prediction accuracy, test overall accuracy, precision improved and AUC is similar.
 
## Documentations

- [Proposal](doc/Proposal.md)  
- [Exploratory data analysis](doc/eda.ipynb)  
- [Final report](doc/final_report.ipynb)

## Usage

To replicate the analysis, clone this GitHub repository, install the dependencies listed below, and run the following commands at the command line/terminal from the root directory of this project:

- Google Chrome browser
```
 python src/dl_xls_to_csv.py --output=credit-default-data.csv
 Rscript src/wrangle.R credit-default-data.csv cleaned-credit-default-data.csv
 python src/eda_plots.py --filepath=data/cleaned-credit-default-data.csv --outdir=results
 python src/analysis.py --input=cleaned-credit-default-data.csv --output=results
```

- Mozilla Firefox browser
```
 python src/dl_xls_to_csv.py --output=credit-default-data.csv
 Rscript src/wrangle.R credit-default-data.csv cleaned-credit-default-data.csv
 python src/eda_plots.py --filepath=data/cleaned-credit-default-data.csv --outdir=results firefox
 python src/analysis.py --input=cleaned-credit-default-data.csv --output=results
```

## Dependencies

  - Web browser(for python Altair to run and save images, see [here](https://altair-viz.github.io/user_guide/saving_charts.html)):
      - A recent version Google Chrome or Mozilla Firefox
      - [Chrome Driver](https://sites.google.com/a/chromium.org/chromedriver/) or [Gecko Driver](https://github.com/mozilla/geckodriver/releases)
  - Python 3.7.3 and Python packages:
      - docopt==0.6.2
      - pandas==0.25.3
      - sklearn==0.22
      - altair==3.2.0
      - selenium==3.141.0
      - numpy==1.17.4
      - imblearn=0.6.1
  - R version 3.6.1 and R packages:
      - tidyverse==1.2.1
      - testthat==2.3.1
      - docopt==0.6.1
      
## References  

Dua, Dheeru, and Casey Graff. 2017. “UCI Machine Learning Repository.” University of California, Irvine, School of Information; Computer Sciences. http://archive.ics.uci.edu/ml.  
Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. Expert Systems with Applications, 36(2), 2473-2480.  


