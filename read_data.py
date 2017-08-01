def get_field_feature(file_name = './test_data.txt'):
    file = open(file_name)
    meta_data = file.readlines()
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

def normalize_data_file(field_dic,feature_dic,meta_file_name = './test_data.txt'):
    file = open('./norm_test_data.txt','w')
    for sample in open(meta_file_name):
        attribute = sample.split()
        str = attribute[0]
        for value in attribute[1:]:
            attrs = value.split(':')
            norm_field = field_dic[attrs[0]]
            norm_feature = feature_dic[attrs[1]]
            str += ' {}:{}:{}'.format(norm_field,norm_feature,attrs[2])
        file.write(str+'\n')
    file.close()
if __name__ == "__main__":
    field_set,feature_set = get_field_feature()
    field_dic = {}
    for idx,item in enumerate(list(field_set)):
        field_dic[item] = idx
    feature_dic = {}
    for idx,item in enumerate(list(feature_set)):
        feature_dic[item] = idx
    normalize_data_file(field_dic,feature_dic)