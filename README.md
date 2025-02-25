# ctDNA_CAT
Data and code for liquid biopsy/cancer associated thromboembolism project. Data presented at [ASH 2023](https://ashpublications.org/blood/article/142/Supplement%201/569/503970/DNA-Liquid-Biopsies-for-Cancer-Associated-Venous) and published in [Nature Medicine 2024](https://www.nature.com/articles/s41591-024-03195-0). Available under [Creative Commons BY-NC-ND 4.0 license](https://creativecommons.org/licenses/by-nc-nd/4.0/deed.en). Additional genomic data for the discovery and validation cohorts is available here: https://www.cbioportal.org/study/summary?id=msk_ctdna_vte_2024 Additional genomic data for the ctDx generalizability cohort can be found here: https://www.cbioportal.org/study/summary?id=nsclc_ctdx_msk_2022

## requirements
python (3.10)
Library dependencies: pandas (1.5.3), numpy (1.24.3), matplotlib (3.7.1), seaborn (0.12.2), statsmodels (0.14.0), lifelines (0.27.7), sksurv (scikit-survival 0.20.0), joblib (optional). () = tested on this version

R 4.2.2 dependencies: cmprsk, survivalROC, data.table, dplyr, randomForestSRC (optional, for run_rsf_cmprsk.R only)

## use instructions
--CompetingRisks.ipynb: R notebook to run 1. Fine-Gray competing risk analyses (example data provided in data folder) and 2. Generate dynamic AUCs from a file of risk scores and gold-standard labels (output of run_rsf_vte.py, provided here as vte_riskscores_validation.csv)

--VTE_visualize.ipynb: python notebook for 1. generating Aalen-Johansen curves as seen in the manuscript (example data provided in the data folder), 2. visualizing outputs from CompetingRisks.ipynb, and 3. using an RSF from run_rsf_vte.py to make inferences on a new dataset

--run_rsf_vte.py: python script to train and validate random survival forest and generates the files vte_rsf_c_index_validation.csv (metrics i.e. c-indices for the model) and vte_riskscores_validation.csv (risk scores per patient). Example here is training on discovery cohort and deploying on prospective validation cohort for a Khorana score+chemotherapy and LB+ model. Saves models as .pkl (NOTE: to run, must make an empty folder within code/ called models/)

--run_saved_rsf_vte.py: python script to load model from .pkl and run on validation dataset (NOTE: to run AFTER run_rsf_vte.py)

--run_rsf_cmprsk.R: R script to train a random survival forest with competing risks (following the example in https://www.randomforestsrc.org/articles/competing.html except substituting in the data from the discovery cohort here)
