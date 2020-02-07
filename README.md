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
- [Final report](https://ubc-mds.github.io/DSCI_522_group-314/doc/final_report.html)  

## Usage


There are two methods to replicate the analysis in this project : 

#### 1. Using Docker

Install [Docker](https://www.docker.com/get-started) and then download/clone this repository. Next, use the command line to navigate to the root of this downloaded/cloned repo andtype the following in the shell:

```
docker run --rm -v C:/DOCs/Canada/BC/UBC/courses/block4/dsci522/lab/DSCI_522_group-314:/home/credit-analysis eribner201/credit-analysis make -C /home/credit-analysis all
```

To reset the repo to a clean state, with no intermediate or results files, run the following command at the command line/terminal from the root directory of this project:

```
docker run --rm -v C:/DOCs/Canada/BC/UBC/courses/block4/dsci522/lab/DSCI_522_group-314:/home/credit-analysis eribner201/credit-analysis make -C /home/credit-analysis clean
```

#### 2. After installing all dependencies (does not depend on Docker)

Clone this GitHub repository, install the dependencies listed below, and run the following commands at the command line/terminal from the root directory of this project:

```
make all
```

To reset the repo to a clean state, with no intermediate or results files, run the following command at the command line/terminal from the root directory of this project:

```
make clean
```

## Dependencies

  - Python 3.7.3 and Python packages:
      - docopt==0.6.2
      - pandas==0.25.3
      - sklearn==0.22
      - altair==3.2.0
      - numpy==1.17.4
      - imblearn=0.6.1
      - seaborn=0.9.0
      - matplotlib=3.1.1
  - R version 3.6.1 and R packages:
      - tidyverse==1.2.1
      - testthat==2.3.1
      - docopt==0.6.1
  - GNU make 4.2.1 
      
## References  

Dua, Dheeru, and Casey Graff. 2017. “UCI Machine Learning Repository.” University of California, Irvine, School of Information; Computer Sciences. http://archive.ics.uci.edu/ml.  
Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. Expert Systems with Applications, 36(2), 2473-2480.  


