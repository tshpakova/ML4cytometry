

import fcsparser
import pandas as pd

"""# example: 10th patient from lean"""

for i in range(10, 11):
  print(i)

  path = 'Carma_lean/Patient ' + str(i) + '_2 (1)/'
  #lympho 
  meta, datalympho = fcsparser.parse(path + 'Lymphocyte.fcs', reformat_meta=True)
  index=list(datalympho.columns)
  datalympho['is_lympho'] = 1

  result = datalympho
  # loop for cd4
  d = {'CD3.fcs': 'is_cd3', 'CD4.fcs': 'is_cd4', 'CD4_CCR7.fcs': 'is_ccr7', 'CD4_CD27.fcs': 'is_cd27', 'CD4_CD28.fcs': 'is_cd28', 'CD4_CD45RA.fcs': 'is_cd45ra',
  'CD4_CD57.fcs': 'is_cd57', 'CD4_CD279.fcs': 'is_cd279', 'CD4_KLRG1.fcs': 'is_KLRG1' }
  for key, val in d.items():
    meta, datacd3 = fcsparser.parse(path+key, reformat_meta=True)
    datacd3[val] = 1
    result = pd.merge(result, datacd3, how='left' ,on= index)   
  index2=list(result.columns)

  # loop for cd8
  d2 = {'CD8.fcs': 'is_cd8', 'CD8_CCR7.fcs': 'is_ccr7', 'CD8_CD27.fcs': 'is_cd27', 'CD8_CD28.fcs': 'is_cd28', 'CD8_CD45RA.fcs': 'is_cd45ra',
  'CD8_CD57.fcs': 'is_cd57', 'CD8_CD279.fcs': 'is_cd279', 'CD8_KLRG1.fcs': 'is_KLRG1' }
  for key, val in d2.items():
    meta, datacd3 = fcsparser.parse(path+key, reformat_meta=True)
    datacd3[val] = 1
    result = pd.merge(result, datacd3, how='left' ,on= index) 


  last_result=result.fillna(0)
  last_result['is_ccr7']=last_result['is_ccr7_x']+last_result['is_ccr7_y']
  last_result['is_cd27']=last_result['is_cd27_x']+last_result['is_cd27_y']
  last_result['is_cd28']=last_result['is_cd28_x']+last_result['is_cd28_y']
  last_result['is_cd45ra']=last_result['is_cd45ra_x']+last_result['is_cd45ra_y']
  last_result['is_cd57']=last_result['is_cd57_x']+last_result['is_cd57_y']
  last_result['is_cd279']=last_result['is_cd279_x']+last_result['is_cd279_y']
  last_result['is_KLRG1']=last_result['is_KLRG1_x']+last_result['is_KLRG1_y']

  index2.append('is_cd8')
  index2[20],index2[27] = index2[27], index2[20]


  real_sampling=last_result[index2]
  save_path = 'Carma_lean/labeld_specimen_0' + str(i) + '_new.csv'
  real_sampling.to_csv (save_path, index = False, header=True)

