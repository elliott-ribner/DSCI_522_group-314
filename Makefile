# Reproduce results and report for this Prediction of Customer Default repo
# author: Zoe Pan
# date: 2020-01-30

#Make all plots, tables and render fianl report as .html format
all: doc/final_report.html

#Make all plots, tables 
results:  results/accuracies.csv results/calssification_report.csv results/confusion_matrix.png results/roc.png results_baseline/calssification_report.csv results_baseline/confusion_matrix.png results_baseline/roc.png results/cat_res_chart.png results/head.csv results/num_describe.csv

#download data and saved in data folder as .csv
data/credit-default-data.csv : src/dl_xls_to_csv.py
	python src/dl_xls_to_csv.py --output=credit-default-data.csv

#data wrangling and save cleaned data to data folder
data/cleaned-credit-default-data.csv : data/credit-default-data.csv src/wrangle.R 
	Rscript src/wrangle.R credit-default-data.csv cleaned-credit-default-data.csv

#save data analysis results to results and results_baseline folder
results/accuracies.csv results/calssification_report.csv results/confusion_matrix.png results/roc.png results_baseline/calssification_report.csv results_baseline/confusion_matrix.png results_baseline/roc.png : data/cleaned-credit-default-data.csv src/analysis.py
	python src/analysis.py --input=cleaned-credit-default-data.csv --output=results

#save exploratory plots and tables to results folder	
results/cat_res_chart.png results/head.csv results/num_describe.csv : data/cleaned-credit-default-data.csv src/eda_plots.py
	python src/eda_plots.py --filepath=data/cleaned-credit-default-data.csv --outdir=results

#create final report .html	
doc/final_report.html : results
	jupyter nbconvert --execute  doc/final_report.ipynb --to html

#clean all data, results
clean:
	rm -f data/*.csv
	rm -f results/*.csv results_baseline/*.csv
	rm -f results/*.png results_baseline/*.png
	rm -f doc/final_report.html