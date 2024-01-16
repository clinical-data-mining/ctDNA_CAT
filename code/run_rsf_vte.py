import numpy as np
import pandas as pd
from sksurv.datasets import load_gbsg2
from sksurv.preprocessing import OneHotEncoder
from sksurv.ensemble import RandomSurvivalForest
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, RepeatedKFold
<<<<<<< HEAD
import pickle
=======
>>>>>>> 1370221c8019c0120723e1519048e4d7d5810c9f

def runRF(df_train,df_tests,cois):

    df_train['vte']=df_train['CAT_DEATH_ENDPT'].astype(int)==1

    X_train = df_train[cois].fillna(df_train[cois].median()).astype(float)
    y_train = df_train[['vte','stop']].apply(tuple, axis=1).values.tolist()
    y_train = np.array(y_train, dtype=[('Status', '?'), ('Survival_in_days', '<f8')])

    rsf_risk_scores_list = []

    random_state = 20
    rsf = RandomSurvivalForest(n_estimators=1000,
                               min_samples_split=10,
                               min_samples_leaf=15,
                               n_jobs=-1,
                               random_state=random_state)
    rsf.fit(X_train, y_train)

    scores = []
    if not type(df_tests)==list:
        df_tests = [df_tests]
    for df_test in df_tests:
        df_test['vte']=df_test['CAT_DEATH_ENDPT'].astype(int)==1
        X_test = df_test[cois].fillna(df_test[cois].median()).astype(float)
        y_test = df_test[['vte','stop']].apply(tuple, axis=1).values.tolist()
        y_test = np.array(y_test, dtype=[('Status', '?'), ('Survival_in_days', '<f8')])

        scores+=[rsf.score(X_test, y_test)]
        rsf_risk_scores = rsf.predict(X_test)
        rsf_risk_scores_list += list(rsf_risk_scores)
<<<<<<< HEAD
=======
        '''try:
            scores+=[rsf.score(X_test, y_test)]
            rsf_risk_scores = rsf.predict(X_test)
            rsf_risk_scores_list += list(rsf_risk_scores)
        except:
            scores+=[np.nan]'''
>>>>>>> 1370221c8019c0120723e1519048e4d7d5810c9f
    return (rsf,scores,rsf_risk_scores_list)


# VARIABLE ASSIGNMENT: columns to add to the model
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
ks_cols = ['KHORANA SCORE','chemotherapy']

# VARIABLE GROUPING: create variable classes for model training (here a Khorana score+chemotherapy model and separate LB+ model)
cois_list = [ks_cols, lbplus_cols]
cois_list_names = ['Khorana Score+chemotherapy','LB+']

# LOAD DATA
vte = pd.read_csv('../data/discovery.csv')
vte2 = pd.read_csv('../data/validation.csv')

# add columns corresponding to significantly altered genes
def queryOncogenicMutations(s, g):
    tok = s.split(';')
    if len(tok)>0:
        return sum([g==t.split(' ')[0] for t in tok])>0
    return False
for g in list(xs_common_genes['Gene']):
    vte[g]=vte['ONCOGENIC_MUTATIONS'].fillna('').apply(lambda x: queryOncogenicMutations(x, g))
    vte2[g]=vte2['ONCOGENIC_MUTATIONS'].fillna('').apply(lambda x: queryOncogenicMutations(x, g))

# RUN RSF (example here is trained on discovery, applied to prospective validation cohort)
scores = pd.DataFrame(columns=cois_list_names)
riskscores = pd.DataFrame()
vte2 = vte2[~vte2['log10(cfDNA concentration)'].isna()]
for j, cois in enumerate(cois_list):
    print(cois_list_names[j])
    jrsf, jscores, jriskscores = runRF(vte,vte2,cois)
    scores.at[0,cois_list_names[j]] = jscores
    riskscores[cois_list_names[j]] = pd.Series(jriskscores)
<<<<<<< HEAD
    # save models
    with open('models/'+cois_list_names[j]+'.pkl', 'wb') as f:
        pickle.dump(jrsf, f)
=======
>>>>>>> 1370221c8019c0120723e1519048e4d7d5810c9f

# OUTPUT (change names to desired output files)
scores.to_csv('vte_rsf_c_index_validation.csv') #c-index scores
riskscores.to_csv('vte_riskscores_validation.csv',index=False) #risk scores per patient
<<<<<<< HEAD
=======

>>>>>>> 1370221c8019c0120723e1519048e4d7d5810c9f
