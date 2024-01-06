import numpy as np
import pandas as pd
from sksurv.datasets import load_gbsg2
from sksurv.preprocessing import OneHotEncoder
from sksurv.ensemble import RandomSurvivalForest
from sklearn.inspection import permutation_importance

xs_mutations = pd.read_csv('data/data_mutations_extended_xs_1_6_23.txt',sep='\t',comment='#')
xs_vc = xs_mutations.Hugo_Symbol.value_counts().head(40)
xs_common_genes = xs_vc[xs_vc>len(xs_vc)*0.05].index
metsitemap = pd.read_csv('metsitemap.txt')
common_cancers = ['Non-Small Cell Lung Cancer','Breast Cancer','Pancreatic Cancer','Melanoma',
          'Prostate Cancer','Bladder Cancer','Esophagogastric Cancer',
          'Hepatobiliary Cancer','Colorectal Cancer','Retinoblastoma','Cancer of Unknown Primary','Histiocytosis','Germ Cell Tumor',
          'Neoplastic Vs Reactive','Endometrial Cancer','Small Cell Lung Cancer','Soft Tissue Sarcoma','Gastrointestinal Stromal Tumor']
demographics_cols = ['AGE','MALE','WHITE', 'ASIAN-FAR EAST/INDIAN SUBCONT', 'BLACK OR AFRICAN AMERICAN']
met_cols = ['N organ sites']+list(metsitemap.SITE_GENIE.unique())
ctDNA_cols = list(xs_common_genes)+['+ctDNA','log10(max VAF)','log10(cfDNA concentration)']
ks_cols = ['KHORANA SCORE','chemotherapy']
ks_component_cols = ['BMI','Platelets','HGB','WBC','chemotherapy']+common_cancers
other_cols = ['lt_start']  #'Albumin'
allcols = ks_component_cols+['KHORANA SCORE']+met_cols+ctDNA_cols+other_cols+demographics_cols

'''cois_list = [ks_cols,
             ks_component_cols,
             met_cols,
             ctDNA_cols,
             common_cancers+demographics_cols+other_cols,
             ['+ctDNA','log10(max VAF)','log10(cfDNA concentration)','chemotherapy']+common_cancers,
             ctDNA_cols+['chemotherapy']+common_cancers,
            allcols]
cois_list_names = ['Khorana Score','KS Components','Metastatic Sites',
                   'Liquid Biopsy','Demographics','LB+ (no genes)','LB+','All']

'''


vte = pd.read_csv('data/xs_VTE_processed.csv')
for c in common_cancers:
    vte[c] = vte['CANCER_TYPE']==c

dftemp = vte[vte['stop']>=0].reset_index()

cois = ctDNA_cols+['chemotherapy']+common_cancers

cois = list(set(cois).intersection(set(dftemp.columns)))

isplit = int(len(dftemp)*0.8)
dftrain = dftemp.iloc[:isplit]
dftest = dftemp.iloc[isplit:]

X_train = dftrain[cois].fillna(dftrain[cois].median()).astype(float)
y_train = dftrain[['dead','stop']].apply(tuple, axis=1).values.tolist()
y_train = np.array(y_train, dtype=[('Status', '?'), ('Survival_in_days', '<f8')])

X_test = dftest[cois].fillna(dftrain[cois].median()).astype(float)
y_test = dftest[['dead','stop']].apply(tuple, axis=1).values.tolist()
y_test = np.array(y_test, dtype=[('Status', '?'), ('Survival_in_days', '<f8')])

rsf = RandomSurvivalForest(n_estimators=1000,
                               min_samples_split=10,
                               min_samples_leaf=15,
                               n_jobs=-1,
                               random_state=42)
rsf.fit(X_train, y_train)
print(f'The c-index of Random Survival Forest is given by {rsf.score(X_test,y_test):.3f}')


from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 20]
# Minimum number of samples required at each leaf node
min_samples_leaf = [5,15,30]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomSurvivalForest()
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 5, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_train, y_train)

print(rf_random.best_params_)

print(
    f"The c-index of random survival forest using these params is "
    f"{rf_random.score(X_test, y_test):.3f}")



