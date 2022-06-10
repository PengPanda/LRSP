import os
import os.path
# path = './vis_feature/test_data/imgs'
# txt = './vis_feature/test_data/test.txt'

class getNames(object):
    def __init__(self,path,folder,txt_name,suffix=None) -> None:
        self.file_path =  os.path.join(path,folder)
        self.txt = path +'/'+ txt_name + '.txt'
        self.suffix = suffix
        if not self.suffix:
            self.suffix = ['jpg','png','jpeg','gif','tiff']

        self.getNames()

    def getNames(self):
        filenames = sorted(self.get_img_list(self.file_path))
        file = open(self.txt,'w')
        for names in filenames:
            name = names.split('.')[0]
            file.write(name + '\n')
            # print(name)
        file.close()
        print('--------### file amount = %d ### getNames done! ---'%len(filenames)) 

    def get_img_list(self,path):
            is_image_file = lambda x : any(x.endswith(extension) 
                                        for extension in self.suffix)
            return [x for x in os.listdir(path) if is_image_file(x)]


if __name__ == '__main__':
    path = '/home/pp/Datasets/FixationDataset/DaytimeDataset/'
    folder = 'imgs/'
    txt_name = 'test'
    suffix = 'png'

    createNames = getNames(path,folder,txt_name,suffix)