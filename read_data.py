import os
import time
def get_field_feature(file_name = './test_data.txt'):
    file = open(file_name)
    meta_data = file.readlines()
    meta_data = meta_data[:2]
    file.close()
    '''
    get fields and features
    '''
    field_set = set()
    feature_set = set()
    for sample in meta_data:
        attribute = sample.split()
        for value in attribute[1:]:
            attrs = value.split(':')
            field_set.add(attrs[0])
            feature_set.add(attrs[1])
    return field_set,feature_set
