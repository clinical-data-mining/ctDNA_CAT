# ctDNA_CAT
Data and code for liquid biopsy/cancer associated thromboembolism project. Note: full genomic data for the discovery and validation cohorts will be available on https://www.cbioportal.org/ upon study release. Genomic data for the ctDx generalizability cohort can be found here: https://www.cbioportal.org/study/summary?id=nsclc_ctdx_msk_2022

## requirements
python (3.10)
Library dependencies: pandas (1.5.3), numpy (1.24.3), matplotlib (3.7.1), seaborn (0.12.2), statsmodels (0.14.0), lifelines (0.27.7), sksurv (scikit-survival 0.20.0). () = tested on this version

R 4.2.2 dependencies: cmprsk, survivalROC, data.table, dplyr

## use instructions
--CompetingRisks.ipynb: R notebook to run Fine-Gray competing risk analyses (example data provided in data folder)

--VTE_visualize.ipynb: python notebook for visualizing outputs from CompetingRisks.ipynb and also for generating Aalen-Johansen curves as seen in the manuscript (example data provided in the data folder)

--run_rsf_vte.py: python script to train and validate random survival forest and generates the files vte_rsf_c_index_validation.csv (metrics i.e. c-indices for the model) and vte_riskscores_validation.csv (risk scores per patient). Example here is training on discovery cohort and deploying on prospective validation cohort for a Khorana score+chemotherapy and LB+ model

--run_rsf_cmprsk.R: R script to train a random survival forest with competing risks (following the example in https://www.randomforestsrc.org/articles/competing.html except substituting in the data from the discovery cohort here)
