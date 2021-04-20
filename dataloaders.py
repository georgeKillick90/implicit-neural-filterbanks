import torch
import cv2


class BirdDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, rootdir, paths, transforms):

        self.rootdir = rootdir
        self.paths = paths
        self.transforms = transforms

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):

        img_meta_data = self.paths[idx][0]

        
        img_category = int(img_meta_data.split(".")[0].split(" ")[1])
        img_meta_data = img_meta_data.split(" ")
        img_meta_data = img_meta_data[1]
        img_path = self.rootdir + img_meta_data

        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

        image = self.transforms(image)


        #################
        #Class labels start at 1 in the csv so we transpose down by 1
        ################

        return image, int(img_category)-1