import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import fitz

from torch.utils.data import Dataset
from tqdm import tqdm

class DocumentDatasetTrain(Dataset):
    def __init__(self, data, labels):
        super().__init__()

        self.data = data
        self.labels = labels

    def __getitem__(self, idx):
        images = self.data[idx]
        labels = self.labels[idx]

        return images, labels
    
    def __len__(self):
        return len(self.data)
    
class DocumentDatasetVal(Dataset):
    def __init__(self, data, labels):
        super().__init__()

        self.data = data
        self.labels = labels

    def __getitem__(self, idx):
        images = self.data[idx]
        labels = self.labels[idx]

        return images, labels
    
    def __len__(self):
        return len(self.data)

class DocumentDataset(Dataset):
    def __init__(self, dataset_path = r"\\10.5.1.36\dataset_IA\dataset_pdf_v1", debug=False) -> None:
        """
        Initializes the DocDataset class.
        Args:
            size (tuple or int): The target size for image standardization.
            blur_kernel (tuple or int): The kernel size to use for image blurring.
            dataset_path (str, optional): The path to the dataset directory. Defaults to r"\\10.5.1.36\dataset_IA\dataset_pdf_v1".
        Attributes:
            dataset_pth (str): Stores the dataset path.
            standardizer (StandardizationIMG): Instance for standardizing images.
            finder (FindsLinesIMG): Instance for detecting lines in images.
            img_path (str): Subdirectory name for images.
            label_path (str): Subdirectory name for labels.
        """
        super().__init__()

        self.dataset_pth = dataset_path
        self.debug = debug

        self.img_path = 'clean_imgs'
        self.label_path = 'labels'

        self.data = []
        self.labels = []

        self.sum_mid = np.zeros(3)
        self.sum_std = np.zeros(3)
        self.count = 0
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        images = self.data[idx]
        labels = self.labels[idx]

        return images, labels
    
    def create_dataset(self):
        """
        Create dataset by loading images and labels from the specified directory.
        Images are resized to 224x224 and normalized. The dataset statistics (mean and std)
        are computed and used to standardize the images.

        Args:
            type_set (str, optional): Type of dataset to create ('uncleaned' or
        """
        labels_dir = os.path.join(self.dataset_pth, self.label_path)
        
        missing_folders = []

        for i, json_file in tqdm(enumerate(os.listdir(labels_dir))):
            json_path = os.path.join(labels_dir, json_file)
            with open(json_path, 'r') as file:
                data = json.load(file)
                ground_truth = json.loads(data['ground_truth'])
                ground_truth = dict(ground_truth)
                ground_truth = ground_truth['tipoDocumento']
                
                if ground_truth not in ['01', '02', '03']:
                    continue
            # Read image
            img_folder_name = json_file.replace(".json", "")
            folder_name = os.path.join(self.dataset_pth, self.img_path, img_folder_name)
            if not os.path.exists(folder_name):
                missing_folders.append(folder_name)
                continue

            num_classes = self.read_imgs(img_folder_name)
            if num_classes > 0:
                self.labels.extend([int(ground_truth) - 1] * num_classes)

        if len(missing_folders) > 0:
            output_txt = "cartelle_mancanti.txt"
            print(f"\nATTENZIONE: Trovate {len(missing_folders)} cartelle mancanti.")
            print(f"Salvataggio lista in: {os.path.abspath(output_txt)}")
            
            with open(output_txt, "w") as f:
                f.write(f"Totale cartelle mancanti: {len(missing_folders)}\n")
                f.write("-" * 30 + "\n")
                for path in missing_folders:
                    f.write(f"{path}\n")

        if len(self.data) > 0:
            self.data = np.array(self.data, dtype=np.float32)
            self.labels = np.array(self.labels, dtype=np.int64)
            
            # compute media and std of dataset
            self.mean = self.data.mean(axis=(0, 1, 2))
            self.std = self.data.std(axis=(0, 1, 2))

            self.data = (self.data - self.mean) / (self.std + 1e-8)
            
    def read_imgs(self, img_folder_name):
        folder_path = os.path.join(f'{self.dataset_pth}{self.img_path}', img_folder_name)

        if not os.path.exists(folder_path):
            print(f"\n[ERRORE PERCORSO] Sto cercando qui: {os.path.abspath(folder_path)}")
            print(f"Ma la cartella non esiste!")
            return 0

        try:
            images_paths = sorted(
                os.path.join(folder_path, f)
                for f in os.listdir(folder_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            )
        except FileNotFoundError:
            return 0

        loaded = 0
        for img_path in images_paths:
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError('Unable to read image')
            self.data.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            loaded += 1

        return loaded
    
dts = DocumentDataset(dataset_path='Dataset/')
dts.create_dataset()