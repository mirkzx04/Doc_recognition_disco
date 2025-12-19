import torch as th
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

import fitz

from ImgCleaner import DynamicDocumentCleaner

def convert_pdf(img_cleaner, pdf_path = 'PDF_Disco/images', ):
    """
    Convert pdf pages into images, then call the cleaner to clean the images and save all of them into "image_document_clean_dataset

    Args : 
        pdf_path (str) : path of PDFs folder
        img_cleaner (ImgCleaner Object) : Class that do image cleaning
        dataset_path (str) : Path of dataset with clean image
    """

    if not os.path.exists(pdf_path):
        return f"Error : {pdf_path} doesn't exist"
    
    # Scan all file into the pdf path
    for file_name in os.listdir(pdf_path):
        if file_name.lower().endswith('.pdf'): # Check if the current file is a pdf
            print(f"Converting..")
            file_path = os.path.join(pdf_path, file_name)

            try:
                doc = fitz.open(file_path)
                base_name = os.path.splitext(file_path)[0] # Return file name without .pdf extension

                # for each pdf pages
                for i, page in enumerate(doc):
                    img_path = f"{file_path[:-4]}_P{i}"

                    mat = fitz.Matrix(2, 2)
                    pix = page.get_pixmap(matrix = mat, alpha = False) # Pixel matrix of i-th page

                    # convert bytebuffer into the numpy array
                    img_array = np.frombuffer(pix.samples, dtype= np.uint8)
                    img_matrix = img_array.reshape(pix.h, pix.w, pix.n)

                    # clean image
                    clean_img = img_cleaner.process(img_matrix, img_path)
                    dim = (224, 224)

                    plt.imshow(cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB))
                    plt.show(block=False) 
                    plt.pause(1) 
                    plt.close()
            except Exception as e:
                print(f"Erro : {e}")

img_cleaner = DynamicDocumentCleaner()
convert_pdf(img_cleaner)