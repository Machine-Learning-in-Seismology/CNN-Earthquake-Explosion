import pandas as pd
from scipy.stats import wilcoxon

themodel = pd.read_csv('results/model5results20210914-172916.csv',delimiter='\t')

themodel_fnr = themodel['fnr']
themodel_fpr = themodel['fpr']

_, themodel_fnr_p = wilcoxon(themodel_fnr)
_, themodel_fpr_p = wilcoxon(themodel_fpr)

linvillemodel = pd.read_csv('results/linville_results20210915-132941.csv',delimiter='\t')

linvillemodel_fnr = linvillemodel['fnr']
linvillemodel_fpr = linvillemodel['fpr']

_, linvillemodel_fnr_p = wilcoxon(linvillemodel_fnr)
_, linvillemodel_fpr_p = wilcoxon(linvillemodel_fpr)

print('Our Model FNR P: {}\nLinville Model FPR P: {}\nOur Model FNR P: {}\nLinville Model FPR P: {}'.format(themodel_fnr_p,linvillemodel_fnr_p,themodel_fpr_p,linvillemodel_fpr_p))

'''
Our Model FNR P: 0.0001308020996998329
Linville Model FPR P: 8.857457687863547e-05
Our Model FNR P: 8.782404130245517e-05
Linville Model FPR P: 8.857457687863547e-05
'''