import os
import os.path
import cv2 
path = './vis_feature/test_data/edges/'

for dirpath, dirnames,filenames in os.walk(path):
    for names in filenames:
        name = names.split('.')[0]
        if names.split('.')[-1] == 'png':
            impath = os.path.join(path,names)
            im = cv2.imread(impath)
            savepath = os.path.join(path,name+'.jpg')
            cv2.imwrite(savepath,im)

            # file.write(name + '\n')
        # print(name)
    # file.close()
print('done!')    
