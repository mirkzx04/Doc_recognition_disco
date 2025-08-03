import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset

from img_preproc.CleanerIMG import CleanerIMG
from img_preproc.FindsLinesIMG import FindsLinesIMG

class DocumentDatasetTrain(Dataset):
    def __init__(self, data, labels):
        super().__init__()

        self.data = data
        self.labels = labels

    def __getitem__(self, idx):
        images = self.data[idx]
        labels = self.labels[idx]

        return images, labels
    
class DocumentDatasetVal(Dataset):
    def __init__(self, data, labels):
        super().__init__()

        self.data = data
        self.labels = labels

    def __getitem__(self, idx):
        images = self.data[idx]
        labels = self.labels[idx]

        return images, labels

class DocumentDataset(Dataset):
    def __init__(self, size, blur_kernel, dataset_path = r"\\10.5.1.36\dataset_IA\dataset_pdf_v1") -> None:
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

        self.cleaner = CleanerIMG(size, blur_kernel)
        self.finder = FindsLinesIMG(low = 10, high=60)

        self.img_path = 'images'
        self.label_path = 'labels'

        self.data = []
        self.labels = []

        self.sum_mid = np.zeros(3)
        self.sum_std = np.zeros(3)
        self.count = 0

    
    def load_dataset(self) -> None:
        """
        Loads and processes the dataset by iterating over all JSON label files in the specified directory.
        For each JSON file found in the label path, this method:
            - Opens and reads the file.
            - Parses the 'ground_truth' field from the JSON content.
            - Extracts the 'CodiceFiscale' (fiscal code) and 'tipoDocumento' (document type) fields.
            - Calls the `read_img` method with the extracted fiscal code and document type.
        If a JSON file cannot be read or parsed, an error message is printed.
        Raises:
            Prints an error message if a JSON file is not found or cannot be processed.
        """
       
        for json_file in os.listdir(f'{self.dataset_pth}\{self.label_path}'):
            # Read json file
            try : 
                with open(f'{self.dataset_pth}\{self.label_path}\{json_file}', 'r') as file:
                    data = json.load(file)
                    data = json.loads(data['ground_truth'])

                    doc_type = data['tipoDocumento']

                    self.read_img(json_file.replace(".json", ""), doc_type)
            except Exception as e:
                print(f'=== JSON NON TROVATO : {e} ===')

        # Standardize image
        mean = self.sum_mid / self.count
        std = np.sqrt(self.sum_std / self.count - self.mean**2)

        self.data = np.ndarray(self.data, dtype=np.float32)
        self.data = (self.data - mean) / std

        self.labels = np.ndarray(self.labels, dtype=np.int32)

    def read_img(self, fiscal_code, doc_type):
        """
        Reads and processes an image corresponding to the given fiscal code and document type.
        Args:
            fiscal_code (str): The fiscal code used to identify the image file.
            doc_type (str): The type of document, used to determine further processing steps.
        Returns:
            numpy.ndarray or None: The processed image as a NumPy array if the image is found and successfully processed; 
            otherwise, returns None.
        """

        # Extract image
        img_path = os.path.join(f'{self.dataset_pth}\{self.img_path}', f'{fiscal_code}.jpg')
        img = cv2.imread(img_path)

        if img is not None:
            # Resize image
            resize_img = self.cleaner.resize_keep_ratio(img) 
            gray_img = self.cleaner.gray_image(resize_img)

            if doc_type == '02':
                # Take three parts of image and insert to data and its labels
                img_crop = self.finder.crop_document(self.cleaner.blur_image(gray_img))
                y0, y1, y2, y3= self.finder.give_tree_img(img_crop)

                # Cut image
                page_1 = gray_img[y0:y1]
                page_2 = gray_img[y1:y2]
                page_3 = gray_img[y2:y3]

                plt.imshow(page_1, cmap='gray')
                plt.axis('off')
                plt.show()

                plt.imshow(page_2, cmap='gray')
                plt.axis('off')
                plt.show()

                plt.imshow(page_3, cmap='gray')
                plt.axis('off')
                plt.show()

                self.data.extend(page_1)
                self.data.extend(page_2)
                self.data.extend(page_3)

                self.labels.extend(doc_type for _ in range(0, 3))

                self.prepare_standardization(page_1)
                self.prepare_standardization(page_2)
                self.prepare_standardization(page_3)

            else:
                self.data.extend(resize_img)
                self.labels.extend(doc_type)

                self.prepare_standardization(resize_img)
        else:
            print('=== IMAGE NOT FOUND ===')
    
    def prepare_standardization(self, img):
        """
        Updates running totals required for image standardization.
        This method accumulates the sum of pixel values and the sum of squared pixel values
        across all images processed, as well as the total pixel count. These statistics are
        used to compute the mean and standard deviation for dataset normalization.
        Args:
            img (numpy.ndarray): The input image array, expected to have shape (H, W, C),
                where H is height, W is width, and C is the number of channels.
        Updates:
            self.sum_mid (numpy.ndarray): Accumulates the sum of pixel values per channel.
            self.sum_std (numpy.ndarray): Accumulates the sum of squared pixel values per channel.
            self.count (int): Accumulates the total number of pixels processed.
        """

        self.sum_mid += img.sum(axis=(0, 1))
        self.sum_std += (img**2).sum(axis=(0,1))
        self.count += img.shape[0] * img.shape[1]

    def split_dataset(self, num_train, num_val):
        train_set = DocumentDatasetTrain(self.data[:num_train], self.labels[:num_train])
        val_set = DocumentDatasetVal(self.data[num_train+1:num_val], self.labels[num_train+1:num_val])

        return train_set, val_set
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        images = self.data[idx]
        labels = self.labels[idx]

        return images, labels
        





    

                    