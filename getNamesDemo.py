import os
import os.path
path = './vis_feature/test_data/imgs'
txt = './vis_feature/test_data/test.txt'
file = open(txt,'w')

for dirpath, dirnames,filenames in os.walk(path):
    for names in filenames:
        name = names.split('.')[0]
        if names.split('.')[-1] == 'jpeg':
            file.write(name + '\n')
        # print(name)
    file.close()
print('done!')    
