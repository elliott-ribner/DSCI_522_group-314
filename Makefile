all:  results/accuracies.csv results/calssification_report.csv results/confusion_matrix.png results/roc.png results_baseline/accuracies.csv results_baseline/calssification_report.csv results_baseline/confusion_matrix.png results_baseline/roc.png results/cat_res_chart.png results/head.csv results/num_describe.csv

data/credit-default-data.csv : src/dl_xls_to_csv.py
	python src/dl_xls_to_csv.py --output=credit-default-data.csv

data/cleaned-credit-default-data.csv : data/credit-default-data.csv src/wrangle.r 
	Rscript src/wrangle.R credit-default-data.csv cleaned-credit-default-data.csv
	
results/accuracies.csv results/calssification_report.csv results/confusion_matrix.png results/roc.png : data/cleaned-credit-default-data.csv src/analysis.py
	python src/analysis.py --input=cleaned-credit-default-data.csv --output=results
	
results_baseline/accuracies.csv results_baseline/calssification_report.csv results_baseline/confusion_matrix.png results_baseline/roc.png : data/cleaned-credit-default-data.csv src/analysis.py
	python src/analysis.py --input=cleaned-credit-default-data.csv --output=results
	
results/cat_res_chart.png results/head.csv results/num_describe.csv : data/cleaned-credit-default-data.csv src/eda_plots.py
	python src/eda_plots.py --filepath=data/cleaned-credit-default-data.csv --outdir=results
	

clean:
	rm -f data/*.csv
	rm -f results/*.csv results_baseline/*.csv
	rm -f results/*.png results_baseline/*.png