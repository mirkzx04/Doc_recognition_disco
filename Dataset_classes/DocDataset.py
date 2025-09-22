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
    def __init__(self, size, blur_kernel, dataset_path = r"\\10.5.1.36\dataset_IA\dataset_pdf_v1", debug=False) -> None:
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

        self.cleaner = CleanerIMG(debug=debug)
        self.finder = FindsLinesIMG(debug=debug)

        self.img_path = 'images'
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
    
    def create_dataset(self, type_set = 'uncleaned'):
        """
        Create dataset by loading images and labels from the specified directory.
        Images are resized to 224x224 and normalized. The dataset statistics (mean and std)
        are computed and used to standardize the images.

        Args:
            type_set (str, optional): Type of dataset to create ('uncleaned' or
        """
        labels_dir = os.path.join(self.dataset_pth, self.label_path)

        for i, json_file in enumerate(os.listdir(labels_dir)):
            json_path = os.path.join(labels_dir, json_file)
            with open(json_path, 'r') as file:
                data = json.load(file)
                data = json.loads(data['ground_truth'])
            
            # Read image
            base_filename = json_file.replace(".json", "")
            if type_set == 'cleaned':
                self.img_path = 'clean_img'
            img = self.read_img(base_filename)

            # Insert image in dataset
            if img is not None:
                if type_set is not 'cleaned':
                    resized_img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
                normalized_img = resized_img.astype(np.float32) / 255.0
                
                self.data.append(normalized_img)
                self.labels.append(int(data['tipoDocumento']))
                
                # Compute statistics for mean and std
                self.sum_mid += np.mean(normalized_img, axis=(0, 1))  # Mean per channel
                self.sum_std += np.std(normalized_img, axis=(0, 1))   # Std per channel
                self.count += 1

        if self.data:
            self.data = np.array(self.data, dtype=np.float32)
            self.labels = np.array(self.labels, dtype=np.int64)
            
            # compute media and std of dataset
            mean = self.sum_mid / self.count
            std = self.sum_std / self.count
            
            for i in range(len(self.data)):
                self.data[i] = (self.data[i] - mean) / (std + 1e-8)  # +1e-8 per evitare divisione per zero
            
    def read_img(self, filename):
        img_path = os.path.join(self.dataset_pth, self.img_path, f"{filename}.jpg")
        img = cv2.imread(img_path)
        return img
    
    def clean_img(self, img, doc_type):
        if img is None:
            return None

        cleaned_img = self.cleaner.pipeline_clean(img, doc_type)
        return cleaned_img

    def create_clean_dataset(self):
        """
        Crea un dataset pulito salvando le immagini processate nella cartella 'clean_img'.
        Ogni immagine mantiene il nome originale, i documenti multipli vengono salvati 
        con suffissi progressivi (_doc1, _doc2, _doc3).
        
        Returns:
            int: Numero di immagini processate con successo
        """
        # Crea la cartella clean_img se non esiste
        clean_img_path = os.path.join(self.dataset_pth, 'clean_img')
        os.makedirs(clean_img_path, exist_ok=True)
        
        processed_count = 0
        labels_dir = os.path.join(self.dataset_pth, self.label_path)
        total_files = len(os.listdir(labels_dir))
        
        print(f"Inizio pulizia dataset - {total_files} file da processare")
        print(f"Cartella output: {clean_img_path}")
        
        for i, json_file in enumerate(os.listdir(labels_dir)):
            try:
                # Leggi il file JSON
                json_path = os.path.join(labels_dir, json_file)
                with open(json_path, 'r') as file:
                    data = json.load(file)
                    data = json.loads(data['ground_truth'])
                    
                doc_type = data['tipoDocumento']
                base_filename = json_file.replace(".json", "")
                
                # Carica l'immagine originale
                original_img = self.read_img(base_filename, doc_type)
                
                if original_img is None:
                    print(f"Impossibile leggere: {base_filename}")
                    continue
                
                print(f"Processando [{i+1}/{total_files}]: {base_filename} (tipo: {doc_type})")
                
                if doc_type == '99':
                    # Gestione documenti multipli
                    self._process_multi_document_for_cleaning(original_img, base_filename, clean_img_path)
                    processed_count += 1
                else:
                    # Gestione documenti singoli
                    success = self._process_single_document_for_cleaning(
                        original_img, base_filename, doc_type, clean_img_path
                    )
                    if success:
                        processed_count += 1
                        
            except Exception as e:
                print(f"Errore processando {json_file}: {str(e)}")
                continue
        
        print(f"Pulizia completata! {processed_count}/{total_files} immagini processate")
        print(f"Immagini salvate in: {clean_img_path}")
        
        return processed_count
    
    def _process_single_document_for_cleaning(self, img, filename, doc_type, output_path):
        """
        Processa un singolo documento per la pulizia.
        
        Args:
            img: Immagine da processare
            filename: Nome del file (senza estensione)
            doc_type: Tipo documento
            output_path: Percorso di output
            
        Returns:
            bool: True se il processo è andato a buon fine
        """
        try:
            # Pulisci l'immagine
            cleaned_img = self.clean_img(img, doc_type)
            
            if cleaned_img is not None:
                # RIDIMENSIONA L'IMMAGINE A 224x224
                resized_img = cv2.resize(cleaned_img, (224, 224), interpolation=cv2.INTER_AREA)

                # Salva l'immagine pulita
                output_filename = f"{filename}.jpg"
                output_filepath = os.path.join(output_path, output_filename)
                
                success = cv2.imwrite(output_filepath, resized_img)
                
                if success:
                    print(f"   Salvata: {output_filename}")
                    return True
                else:
                    print(f"   Errore nel salvare: {output_filename}")
                    return False
            else:
                print(f"   Immagine scartata per bassa qualità: {filename}")
                return False
                
        except Exception as e:
            print(f"   Errore processando {filename}: {str(e)}")
            return False
    
    def _process_multi_document_for_cleaning(self, img, filename, output_path):
        """
        Processa documenti multipli per la pulizia, salvando ogni documento separatamente.
        
        Args:
            img: Immagine contenente documenti multipli
            filename: Nome del file base (senza estensione)
            output_path: Percorso di output
        """
        try:
            # Usa process_multi_documents per separare i documenti
            multi_docs = self.cleaner.process_multi_documents(img, self.finder)
            
            if not multi_docs:
                print(f"   Nessun documento estratto da: {filename}")
                return
            
            # Salva ogni documento estratto
            for i, doc in enumerate(multi_docs):
                if doc is not None:
                    # Nome con suffisso progressivo
                    output_filename = f"{filename}_doc{i+1}.jpg"
                    output_filepath = os.path.join(output_path, output_filename)
                    
                    success = cv2.imwrite(output_filepath, doc)
                    
                    if success:
                        print(f"   Estratto e salvato: {output_filename}")
                    else:
                        print(f" Errore nel salvare: {output_filename}")
                        
        except Exception as e:
            print(f"Errore processando documenti multipli {filename}: {str(e)}")
    
    def split_dataset(self, train_ratio, val_ratio):
        """
        Splits the dataset into training and validation sets based on the provided ratios.
        
        Args:
            train_ratio (float): Proportion for the training set 
            val_ratio (float): Proportion for the validation set 

        Returns:
            tuple: (train_dataset, val_dataset) - Instances of DocumentDatasetTrain and DocumentDatasetVal
        """
        if not self.data or not self.labels:
            raise ValueError("Empty dataset! Please run create_dataset() first.")
            
        if abs(train_ratio + val_ratio - 1.0) > 1e-6:
            raise ValueError("train_ratio + val_ratio must equal 1.0")

        # Compute the number of samples
        total_samples = len(self.data)
        num_train = int(total_samples * train_ratio)
        num_val = total_samples - num_train
        
        print(f"Splitting dataset:")
        print(f"  Total samples: {total_samples}")
        print(f"  Training set: {num_train} samples ({train_ratio*100:.1f}%)")
        print(f"  Validation set: {num_val} samples ({val_ratio*100:.1f}%)")

        #  Create random indices to shuffle the dataset
        indices = np.random.permutation(total_samples)
        

        train_indices = indices[:num_train]
        val_indices = indices[num_train:]

        # Extract data for training and validation
        train_data = self.data[train_indices]
        train_labels = self.labels[train_indices]
        
        val_data = self.data[val_indices]
        val_labels = self.labels[val_indices]

        # Create the datasets
        train_dataset = DocumentDatasetTrain(train_data, train_labels)
        val_dataset = DocumentDatasetVal(val_data, val_labels)
        
        return train_dataset, val_dataset