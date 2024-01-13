# ctDNA_CAT
Data and code for liquid biopsy/cancer associated thromboembolism project

## requirements
python (3.10)
Library dependencies: pandas (1.5.3), numpy (1.24.3), matplotlib (3.7.1), seaborn (0.12.2), statsmodels (0.14.0), lifelines (0.27.7), sksurv (scikit-survival 0.20.0). () = tested on this version

R 4.2.2 dependencies: cmprsk, survivalROC, data.table, dplyr

## use instructions
--CompetingRisks.ipynb: R notebook to run Fine-Gray competing risk analyses (example data provided in data folder)

--VTE_visualize.ipynb: python notebook for visualizing outputs from CompetingRisks.ipynb and also for generating Aalen-Johansen curves as seen in the manuscript (example data provided in the data folder)

--run_rsf_vte.py: python script to train and validate random survival forest (note: not linked to example data, issue pending)
