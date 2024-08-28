import numpy as np
import pandas as pd
import joblib

loaded_rsf = joblib.load('models/LB+.pkl')

# Use the loaded model for predictions
vte2 = pd.read_csv('../data/validation.csv')

xs_common_genes = pd.read_csv('common_gene_list.txt')
common_cancers = ['Non-Small Cell Lung Cancer',
 'Breast Cancer',
 'Pancreatic Cancer',
 'Melanoma',
 'Prostate Cancer',
 'Bladder Cancer',
 'Esophagogastric Cancer',
 'Hepatobiliary Cancer',
 'Colorectal Cancer']

lbplus_cols = list(xs_common_genes['Gene'])+['+ctDNA','log10(max VAF)','log10(cfDNA concentration)']+common_cancers+['chemotherapy']

X = vte2[lbplus_cols]

chf = loaded_rsf.predict_cumulative_hazard_function(X)

# Calculate probability of an event within 180 days
time_horizon = 180
event_probabilities = []

for individual_chf in chf:
    chf_at_time_horizon = individual_chf(time_horizon)
    survival_prob = np.exp(-chf_at_time_horizon)
    event_prob = 1 - survival_prob
    event_probabilities.append(event_prob)

print(event_probabilities)
