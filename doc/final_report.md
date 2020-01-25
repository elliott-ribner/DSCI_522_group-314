#My Paper

Depndincies
```bash
sudo apt-get install texlive-latex-extra
sudo apt-get install texlive-bibtex-extra
```

Machine learning is used to in finger print images classification


```python
import pandas as pd
from sklearn.model_selection import train_test_split
import altair as alt
alt.data_transformers.enable('json')
#alt.renderers.enable('notebook')
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np
from sklearn.metrics import accuracy_score, plot_confusion_matrix, confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.metrics import recall_score, precision_score
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from docopt import docopt
from sklearn.feature_selection import RFECV
```


```python
evaluation_matrix = pd.read_csv("../results/accuracies.csv")
evaluation_matrix_base = pd.read_csv("../results_baseline//accuracies.csv")

head = pd.read_csv("../results/head.csv")
summary=pd.read_csv("../results/num_describe.csv")

test_accuracy = round(evaluation_matrix.iloc[0,1],2)
test_recall=round(evaluation_matrix.iloc[2,1],2)
precision_accuracy=round(evaluation_matrix.iloc[3,1],2)
auc=round(evaluation_matrix.iloc[4,1],2)

evaluation_matrix;

```

# **Table of Content:**
* [Summary](#first-bullet)
* [Introduction](#second-bullet)
* [Methods](#third-bullet)
* [Results](#fourth-bullet)
* [Conclusions](#fifth-bullet)
* [References](#ref)

# 1. Summary <a class="anchor" id="first-bullet"></a>


In this project we try to find the best features that best predict default customers using machine learning tools. `Logestic Regression` was found to achieve acceptable results on the test data provided to the trained model. The accuracy of the model on test data was about {{test_accuracy}} and the recall on test data found to be {{test_recall}}. The precision for the model on the test was about {{precision_accuracy}} .The area under the ROC Curve for the final model is {{auc}}.

Due to the risk associated with wrongly labeled customers as non-defaul the model was designed to reduce the false positive (false postive rate). This was also balanced with the overall accuracy on the training data. The model predict the following {{}} features to be the most important features to predict customers default.

1. Amount of the given credit (NT dollar): it includes both the individual consumer credit and his/her family (supplementary) credit.
2. EDUCATION
3. MARRIAGE
4. AGE
5. Past monthly repayment status in September 2005
6. Past monthly repayment status in September 2005
7. Amount of previous payment (NT dollar) in September 2005



# 2. Introduction <a class="anchor" id="second-bullet"></a>
Prediction of customers defaul behavior is critically important in Risk Management. In pariticular, anticipating features associated with the highest prediction power may reduce the overall lender's credit risk. In this study we perform data analysis to learn features that predict default payment.


# 3. Methods <a class="anchor" id="third-bullet"></a>
## Data
We used data from the Taiwanese market in 2005. The Data Set is available from [UCI Machine Learning Repository Irvine, CA: University of California, School of Information and Computer Science](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients). It is a collection of information
containing 23 features from 30,000 customers. The data was originally publicized by Chung Hua University of Taiwan and Tamkang University of Taiwan. Features include :

- `LIMIT_BAL`: Amount of the given credit (NT dollar): it includes both the individual consumer credit and his/her family (supplementary) credit. 
- `SEX`: Gender(1 = male; 2 = female).
- `EDUCATION`: Education (1 = graduate school; 2 = university; 3 = high school; 4 = others). 
- `MARRIAGE`: Marital status (1 = married; 2 = single; 3 = others).  
- `AGE`: Age (year).  
- `PAY_1`, `PAY_2`, ..., `PAY_6`: Past monthly repayment status in September 2005, August 2005, ..., April 2005 respectively. ( -1 = pay duly; 1 = payment delay for one month; 2 = payment delay for two months; . . .; 8 = payment delay for eight months; 9 = payment delay for nine months and above.)  
- `BILL_AMT1`, `BILL_AMT2`, ..., `BILL_AMT6`: Amount of bill statement (NT dollar) in September 2005, August 2005, ..., April 2005 respectively.  
- `PAY_AMT1`, `PAY_AMT2`, ..., `PAY_AMT6`: Amount of previous payment (NT dollar) in September 2005, August 2005, ..., April 2005 respectively.  



## Analysis

Immediately after importing the data it was split into traning and test data. Only 75% of the data was used to train the models and the test data was only used to obtain the test performance of the model on unseen data. 

{{head}}

Figure 1. Head of the data used in this study. 

Next, we created list for numeric and categorical features, below is the summary of the traning data. It shows that that mean, standard deviation, min, max etc. The bill amount, payment amount and credit limit ranges are roughly similar which are around 800,000. It's interesting that The medians for the bill statement amounts are around 20,000, but the medians for payment amounts are 2,000. Age ranges from 21 to 75 which is reasonable.

{{summary}}

Figure 2. Summary the data used in this study. 

To learn the association between numeric features we explored their inter-correlations which can be seen below. 
We can observe that some features a stronger co-linearity such as BILL-AMT1,BILL-AMT2,.. to BILL-AMT6. 

![](../results/num_corr_chart.png)


Figure 3. Inter-correlation between features

We can also study the correlation between the features and the response varibale. We can see that some of the features have stronger correlation with the response varibale than others, for example LIMIT_BALANCE and Age.

![](../results/num_res_chart.png)


[](roc.png)

Figure 4. Correlation between numeric features and response

# 4. Results <a class="anchor" id="fourth-bullet"></a>


We selected `LogisticRegression` as our model since it is more robus given that the dataset has many of the features are not normally distributed. In addition to that fact that is mucher interpetable than more complex models

We started the analysis by applying a robust scalar on the traning data-set as most of features are not normally distributed and due to the high amount of outliers. Since the EDA analysis revealed that many features have strong multio-colinearity we used recursive feature elimination `RFE` to identify the most useful predictors. Then we dropped those columns that are deemed as less useful.

The hyper parameters `C` was tunned in the range from -4 to 20 using 5-fold cross validation and the model was then fitted with the best hyper paramter. Let us now look at the result by glancing into the confusion matrix.

![](../results/confusion_matrix.png)

Figure 5. Confusion matrix of the fitted model with 7 features

We can see that the best model which uses 7 features tends to correctly predict the customer that defualt better than the base case model which use all the features. This is critically importnt in risk management. 

![](../results_baseline/confusion_matrix.png)

Figure 6. Confusion matrix of the fitted model with all 23 features

In terms of accuracy the results are shown below, we can see that the accuracy of the model on test data was about {{test_accuracy}} and the recall on test data found to be {{test_recall}}. The precision for the model on the test was about {{precision_accuracy}} .The area under the ROC Curve for the final model is {{auc}}.



```python
evaluation_matrix
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>result</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>test accuracy</td>
      <td>0.740933</td>
    </tr>
    <tr>
      <th>1</th>
      <td>train accuracy</td>
      <td>0.741733</td>
    </tr>
    <tr>
      <th>2</th>
      <td>test recall</td>
      <td>0.567372</td>
    </tr>
    <tr>
      <th>3</th>
      <td>test precision</td>
      <td>0.433518</td>
    </tr>
    <tr>
      <th>4</th>
      <td>auc score</td>
      <td>0.707454</td>
    </tr>
  </tbody>
</table>
</div>



This is also a good improvement over the base model which use all the 23 features as can see below


```python
evaluation_matrix_base
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>result</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>test accuracy</td>
      <td>0.676267</td>
    </tr>
    <tr>
      <th>1</th>
      <td>train accuracy</td>
      <td>0.677244</td>
    </tr>
    <tr>
      <th>2</th>
      <td>test recall</td>
      <td>0.656798</td>
    </tr>
    <tr>
      <th>3</th>
      <td>test precision</td>
      <td>0.368850</td>
    </tr>
    <tr>
      <th>4</th>
      <td>auc score</td>
      <td>0.721871</td>
    </tr>
  </tbody>
</table>
</div>



ROC was plotted to to measure the model's discriminative ability. We can see that the model perform fairly good compared with the base model. 

![](../results/roc.png)

Figure 7. ROC curve for the fitted model with 7 features

# 4. Conclusions <a class="anchor" id="fifth-bullet"></a>

We were able to successfully use `LogisticRegression` model to find the most important features that predict customer default. The model acheives an acceptable level of accuracy on the testing data, better tunning of hyper paramters may result a higher accuracy. Following are the top 7 features :

1. Amount of the given credit (NT dollar): it includes both the individual consumer credit and his/her family (supplementary) credit.
2. EDUCATION
3. MARRIAGE
4. AGE
5. Past monthly repayment status in September 2005
6. Past monthly repayment status in September 2005
7. Amount of previous payment (NT dollar) in September 2005


## References <a class="anchor" id="ref"></a>

[1] Dheeru Dua and Casey Graff. UCI machine learning repository, 2017.

[2] Guido Van Rossum and Fred L. Drake. Python 3 Reference Manual. CreateSpace, Scotts Valley, CA, 2009

<cite data-cite="Python"></cite>
<cite data-cite="Dua:2019"></cite>


