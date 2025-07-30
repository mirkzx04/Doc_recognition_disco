import os
import cv2
import numpy as np

def load_dataset(dataset_path = './dataset_pdf_v1/images'):
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

            if img is not None:
                imgs.append(imgs)
        
        tst += 1

        if tst == 5:
            break

    return np.array(imgs)

if __name__ == "__main__":
    image = load_dataset()
    
    

