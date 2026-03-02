import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import fitz

from tqdm import tqdm

import torch as th
from torch.utils.data import DataLoader, random_split

import pytorch_lightning as plt
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from ImgCleaner import DynamicDocumentCleaner 

from Training.LitModule import LitModule

from Dataset_classes.DocDataset import DocumentDataset

def clean_dataset(img_matrix, img_path, img_cleaner):
    clean_img = img_cleaner.process(img_matrix, img_path)
    dim = (224, 224)
    return cv2.resize(clean_img, dim, interpolation=cv2.INTER_AREA)

def convert_pdf(img_cleaner, pdf_path = 'PDF_Disco/images', output_root = 'clean_imgs'):
    """
    Convert pdf pages into images, then call the cleaner to clean the images and save all of them into "image_document_clean_dataset

    Args : 
        pdf_path (str) : path of PDFs folder
        img_cleaner (ImgCleaner Object) : Class that do image cleaning
        dataset_path (str) : Path of dataset with clean image
    """

    if not os.path.exists(pdf_path):
        return f"Error : {pdf_path} doesn't exist"
    
    if not os.path.exists(output_root):
        os.makedirs(output_root, exist_ok=True)
    
    # Scan all file into the pdf path
    for file_name in tqdm(os.listdir(pdf_path)):
        if file_name.lower().endswith('.pdf'): # Check if the current file is a pdf
            file_path = os.path.join(pdf_path, file_name)

            try:
                doc = fitz.open(file_path)
                doc_name = os.path.splitext(file_path)[0] # Return file name without .pdf extension

                doc_folder = os.path.join(output_root, doc_name)
                os.makedirs(doc_folder, exist_ok=True)

                # for each pdf pages
                for i, page in enumerate(doc):
                    img_path = f"{file_path[:-4]}_P{i}"

                    mat = fitz.Matrix(2, 2)
                    pix = page.get_pixmap(matrix = mat, alpha = False) # Pixel matrix of i-th page

                    # convert bytebuffer into the numpy array
                    img_array = np.frombuffer(pix.samples, dtype= np.uint8)
                    img_matrix = img_array.reshape(pix.h, pix.w, pix.n)

                    # clean image
                    clean_img = clean_dataset(img_matrix, img_path, img_cleaner)

                    save_path = os.path.join(doc_folder, f'page_{i}.png')
                    cv2.imwrite(save_path, clean_img)
            except Exception as e:
                print(f"Erro : {e}")

print('---- Cleaning')
img_cleaner = DynamicDocumentCleaner()
convert_pdf(img_cleaner)

# batch_size = 128

# dataset = DocumentDataset(dataset_path='Dataset/')
# dataset_size = len(dataset)
# print(f'---- Dataset len : {len(dataset)} ----')

# train_size = int(0.8 * dataset_size)
# val_size = int(dataset_size - train_size)

# generator = th.Generator().manual_seed(42)
# train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)

# train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(dataset=val_set, batch_size = True, shuffle=False)
