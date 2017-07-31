# coding: utf-8
for c in data.columns:
    try:
        data[c] = data[c].apply(lambda x: x.replace('/home/daniel/deep-learning/sdc/Car-Behavioral-Cloning/data/', ''))
    except:
        pass
