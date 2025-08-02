import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

dataset_path = r"\\10.5.1.36\dataset_IA\dataset_pdf_Solo_Rinnovi\images"

from Dataset_classes.DocDataset import DocumentDataset

def load_dataset(standardizer, dataset_path = './dataset_pdf_v1/images'):
    """
    Loading document image
    """
    imgs = []
    tst = 0

    for filename in os.listdir(dataset_path):
        # Check file extension
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):

            # Read image with Open-CV
            img_path = os.path.join(dataset_path, filename)
            img = cv2.imread(img_path)
            img_resize = standardizer.resize_keep_ratio(img)

            if img is not None:
                imgs.append(img_resize)
        
        tst += 1

        if tst == 2:
            return np.array(imgs)

    return np.array(imgs)

if __name__ == "__main__":
    # Load documentation dataset
    doc_dataset = DocumentDataset(size=(5000, 5000), blur_kernel=(4,4))
    doc_dataset.load_dataset()


    
    

