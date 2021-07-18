import cv2 as cv
import os
import numpy as np

class CvUtil:
    
    def resize(self, width, height, image):
        return cv.resize(image, (width, height), interpolation=self.inter)
    

    def data_loader(self, width, height, imagePaths):
        data = []
        labels = []
        for path in imagePaths:
            image = cv.imread(path)
            label = path.split(os.path.sep)[-2]

            #preprocessor
            image = self.resize(width, height, image)

            data.append(image)
            labels.append(label)
        
        return (np.array(data), np.array(labels))


