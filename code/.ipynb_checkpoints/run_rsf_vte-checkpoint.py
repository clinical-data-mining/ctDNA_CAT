import numpy as np
import pandas as pd
from sksurv.datasets import load_gbsg2
from sksurv.preprocessing import OneHotEncoder
from sksurv.ensemble import RandomSurvivalForest
from sklearn.inspection import permutation_importance

def runRF(df_train,df_tests,cois):

    X_train = df_train[cois].fillna(df_train[cois].median()).astype(float)
    y_train = df_train[['dead','stop']].apply(tuple, axis=1).values.tolist()
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
        X_test = df_test[cois].fillna(df_test[cois].median()).astype(float)
        y_test = df_test[['dead','stop']].apply(tuple, axis=1).values.tolist()
        y_test = np.array(y_test, dtype=[('Status', '?'), ('Survival_in_days', '<f8')])
        try:
            scores+=[rsf.score(X_test, y_test)]
            rsf_risk_scores = rsf.predict(X_test)
            rsf_risk_scores_list += list(rsf_risk_scores)
        except:
            scores+=[np.nan]
    return (rsf,scores,rsf_risk_scores_list)

xs_mutations = pd.read_csv('../data/data_mutations_extended_xs_1_6_23.txt',sep='\t',comment='#')
xs_vc = xs_mutations.Hugo_Symbol.value_counts().head(40)
xs_common_genes = xs_vc[xs_vc>len(xs_vc)*0.05].index
metsitemap = pd.read_csv('../metsitemap.txt')
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
#ks_ctDNA_simple_cols = ks_component_cols+['KHORANA SCORE','lt_start']+met_cols+common_cancers+['+ctDNA','log10(max VAF)','log10(cfDNA concentration)']
allcols = ks_component_cols+['KHORANA SCORE']+met_cols+ctDNA_cols+other_cols+demographics_cols



cois_list = [ks_cols,
             ks_component_cols,
             met_cols,
             ctDNA_cols,
             common_cancers+demographics_cols+other_cols,
             ['+ctDNA','log10(max VAF)','log10(cfDNA concentration)','chemotherapy']+common_cancers,
             ctDNA_cols+['chemotherapy']+common_cancers,
            allcols]
cois_list_names = ['Khorana Score','KS Components','Metastatic Sites',
                   'Liquid Biopsy','Demographics','LB+ (no genes)','LB+','All']


from sklearn.model_selection import KFold, RepeatedKFold

vte = pd.read_csv('../data/xs_VTE_processed.csv')
resbio_VTE = pd.read_csv('../data/resbio_VTE_processed.csv')
vte2 = pd.read_csv('../data/vte2_main.csv')
vte2['Platelets'] = vte2['Platelets.']

for c in common_cancers:
    resbio_VTE[c] = False
resbio_VTE['Non-Small Cell Lung Cancer']=True

for g in xs_common_genes:
    vte2[g] = vte2['ONCOGENIC_MUTATIONS'].fillna('').str.contains(g)

for c in common_cancers:
    vte[c] = vte['CANCER_TYPE']==c
    vte2[c] = vte2['CANCER_TYPE']==c

dftemp = vte[(vte['stop']>=0)].reset_index()

scores = pd.DataFrame(columns=cois_list_names)
riskscores = pd.DataFrame()
vte2 = vte2[~vte2['log10(cfDNA concentration)'].isna()]
for j, cois in enumerate(cois_list):
    print(cois_list_names[j])
    jrsf, jscores, jriskscores = runRF(vte,vte2,list(set(cois).intersection(set(dftemp.columns))))
    scores.at[0,cois_list_names[j]] = jscores
    riskscores[cois_list_names[j]] = pd.Series(jriskscores)

scores.to_csv('vte_rsf_scores_vte2_validation.csv')
riskscores.to_csv('vte_riskscores_vte2_val.csv',index=False)

'''
kf = KFold(n_splits=5)# Define the split - into n_splits folds
kf.get_n_splits(dftemp) # returns the number of splitting iterations in the cross-validator

scores = pd.DataFrame(columns=cois_list_names,index=range(5),dtype=object)

for j, cois in enumerate(cois_list):
    print(cois_list_names[j])
    for i, (train_index, test_index) in enumerate(kf.split(dftemp)):
        print(f'fold: {i}')
        train = dftemp.loc[train_index]
        # test = [dftemp.loc[test_index]]
        test = dftemp.loc[test_index]
        test = [test[test.new_chemotherapy]] #*** mask for test subset
        for c in common_cancers[:9]:
            test += [dftemp.loc[test_index][dftemp.loc[test_index].CANCER_TYPE==c]]
        scores.at[i,cois_list_names[j]]=runRF(train,test,
                list(set(cois).intersection(set(dftemp.columns))))[1] #?[0][1]
scores.to_csv('vte_rsf_scores_newchemotherapy.csv')

resbio_VTE['Kingfisher']=resbio_VTE['Kingfisher'].fillna(False)
resbio_VTE.loc[resbio_VTE.Kingfisher,'log10(cfDNA concentration)'] = resbio_VTE[resbio_VTE.Kingfisher]['log10(cfDNA concentration)']*0.59+0.09
resbio_VTE.loc[~resbio_VTE.Kingfisher,'log10(cfDNA concentration)'] = resbio_VTE[~resbio_VTE.Kingfisher]['log10(cfDNA concentration)']*0.46+0.29
scores_finalval = pd.Series()
for j, cois in enumerate(cois_list):
    print(cois_list_names[j])
    scores_finalval.loc[cois_list_names[j]]=runRF(vte[vte['stop']>=0],resbio_VTE,
            list(set(cois).intersection(set(dftemp.columns))))[1][0] #?[0][1]

scores_finalval.to_csv('vte_rsf_scores_resbio_adjcfdna.csv')

# add sydney
sydney = pd.read_csv('data/sydney_processed.csv')

sydney['dead']=~sydney['Date of VTE'].isna()
sydney['MRN'] = sydney['Resolution Patient ID']
sydney['log10(max VAF)'] = sydney['max_vaf']
sydney['Kingfisher']=sydney['Assay']=='ctDx Lung 8'
sydney['Kingfisher']=sydney['Kingfisher'].fillna(False)
sydney.loc[sydney.Kingfisher,'log10(cfDNA concentration)'] = sydney[sydney.Kingfisher]['log10(cfDNA concentration)']*0.59+0.09
sydney.loc[~sydney.Kingfisher,'log10(cfDNA concentration)'] = sydney[~sydney.Kingfisher]['log10(cfDNA concentration)']*0.46+0.29

sydney_unfiltered = pd.read_csv('data/sydney_5_19_23.csv')
sydney_unfiltered['metastatic_datetime']=pd.to_datetime(sydney_unfiltered['metastatic_datetime'])
sydney_unfiltered['resbio_datetime']=pd.to_datetime(sydney_unfiltered['resbio_datetime'])
sydney_unfiltered['lt_start']=(sydney_unfiltered['resbio_datetime']-sydney_unfiltered['metastatic_datetime']).dt.days

sydney = pd.merge(sydney,sydney_unfiltered[['MRN','lt_start']],on='MRN')

cois_list = [ctDNA_cols,
             common_cancers+demographics_cols+other_cols,
             ['+ctDNA','log10(max VAF)','log10(cfDNA concentration)','chemotherapy']+common_cancers,
             ctDNA_cols+['chemotherapy']+common_cancers]
cois_list_names = ['Liquid Biopsy','Demographics','LB+ (no genes)','LB+']

scores_finalval = pd.Series()
for j, cois in enumerate(cois_list):
    print(cois_list_names[j])
    scores_finalval.loc[cois_list_names[j]]=runRF(vte[vte['stop']>=0],sydney,
            list(set(cois).intersection(set(sydney.columns))))[1][0] #?[0][1]

scores_finalval.to_csv('vte_rsf_scores_sydney_adjcfdna.csv')
'''
