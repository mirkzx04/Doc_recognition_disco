import os
import json
import cv2

from torch.utils.data import Dataset

from img_preproc.StandardizationIMG import StandardizationIMG
from img_preproc.FindsLinesIMG import FindsLinesIMG

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

        self.dataset_pth = dataset_path

        self.standardizer = StandardizationIMG(size, blur_kernel)
        self.finder = FindsLinesIMG(low = 10, high=60, kernel_size=(5 , 5))

        self.img_path = 'images'
        self.label_path = 'labels'

        self.data = []
        self.labels = []

    
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

                    fiscal_code = data['CodiceFiscale']
                    doc_type = data['tipoDocumento']

                    self.read_img(fiscal_code, doc_type)
            except:
                print('=== JSON NON TROVATO ===')

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
        img_path = os.path.join(f'{self.dataset_pth}\{self.img_path}', fiscal_code)
        img = cv2.imread(img_path)

        if img is not None:
            # Resize image
            resize_img = self.standardizer.resize_keep_ratio(img)

            if doc_type == '02':
                # Take three parts of image and insert to data and its labels
                page_1, page_2, page_3 = self.finder.give_tree_img(self.standardizer.blurrer(resize_img))
                self.data.extend(page_1)
                self.data.extend(page_2)
                self.data.extend(page_3)

                self.labels.extend(doc_type for _ in range(0, 2))
            else:
                self.data.extend(resize_img)
                self.labels.extend(doc_type)





    

                    