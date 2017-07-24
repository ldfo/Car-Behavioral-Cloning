# coding: utf-8
for c in data.columns:
    data[c] = data[c].apply(lambda x: x.replace('/home/daniel/deep-learning/sdc/sdc_training_data/', 'data/'))
    
