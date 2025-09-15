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
        self.finder = FindsLinesIMG(low=10, high=60)
        
        if debug:
            print("🐛 DEBUG MODE ATTIVATO nel DocumentDataset")
            print(f"📁 Dataset path: {dataset_path}")
            print(f"📐 Target size: {size}")
            print(f"🌀 Blur kernel: {blur_kernel}")

        self.img_path = '/images'
        self.label_path = '/labels'

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
       
        for json_file in os.listdir(f'{self.dataset_pth}{self.label_path}'):
            # Read json file
            with open(f'{self.dataset_pth}{self.label_path}/{json_file}', 'r') as file:
                data = json.load(file)
                data = json.loads(data['ground_truth'])

                doc_type = data['tipoDocumento']

                original_img = self.read_img(json_file.replace(".json", ""), doc_type)
                if original_img is not None:
                    # Gestione speciale per documenti multipli (classe 3)
                    if doc_type == '03':
                        # Usa process_multi_documents per dividere l'immagine
                        multi_docs = self.cleaner.process_multi_documents(original_img, self.finder)
                        
                        for i, doc in enumerate(multi_docs):
                            if doc is not None:
                                # Ogni documento estratto diventa un campione separato
                                self.data.append(doc)
                                self.labels.append(2)  # Label per classe multipli
                                print(f"Documento {i+1}/3 estratto da immagine multipla")
                                
                    else:
                        # Gestione normale per tessere e fogli singoli
                        cleaned_img = self.clean_img(original_img, doc_type)
                        
                        if cleaned_img is not None:
                            # Aggiungi l'immagine pulita al dataset
                            self.data.append(cleaned_img)
                            # Mappa il tipo documento a label numerica
                            label_map = {'01': 0, '02': 1}  # tessera, foglio
                            self.labels.append(label_map.get(doc_type, 0))
                            print(f"Immagine processata con successo - Tipo: {doc_type}")
                        else:
                            print(f"Immagine scartata durante la pulizia - Tipo: {doc_type}")
                else:
                    print(f"Impossibile leggere immagine: {json_file}")

        # Standardize image
        self.data = np.array(self.data, dtype=np.float32)
        mean = self.data.mean(); std = self.data.std()
        self.data = (self.data - mean) / std

        self.labels = np.array(self.labels, dtype=np.int32)

    def read_img(self, fiscal_code, doc_type):
        img_path = os.path.join(f'{self.dataset_pth}{self.img_path}', f'{fiscal_code}.jpg')
        return cv2.imread(img_path)

    def clean_img(self, img, doc_type='01'):
        """
        Pipeline completa di pulizia per documenti utilizzando CleanerIMG professionale.
        
        Args:
            img: Immagine da pulire
            doc_type: Tipo documento ('01'=tessera, '02'=foglio, '03'=multipli)
        
        Returns:
            Immagine pulita e elaborata
        """
        if img is None:
            print("Immagine non valida")
            return None
            
        print(f"Inizio pulizia documento tipo: {doc_type}")
        
        # Usa la pipeline completa di pulizia
        cleaned_img = self.cleaner.pipeline_clean(img, doc_type)
        
        # Valuta la qualità dell'immagine risultante
        quality_scores = self.cleaner.assess_quality(cleaned_img)
        print(f"Qualità immagine: {quality_scores['overall']:.2f}")
        print(f"   - Nitidezza: {quality_scores['sharpness']:.2f}")
        print(f"   - Contrasto: {quality_scores['contrast']:.2f}")
        print(f"   - Luminosità: {quality_scores['brightness']:.2f}")
        
        # Filtra immagini di qualità troppo bassa
        if quality_scores['overall'] < 0.3:
            print("Immagine scartata per bassa qualità")
            return None
            
        # Debugging opzionale
        if hasattr(self, 'debug') and self.debug:
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title('Originale')
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(cv2.cvtColor(cleaned_img, cv2.COLOR_BGR2RGB))
            plt.title('Pulita')
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.bar(quality_scores.keys(), quality_scores.values())
            plt.title('Metriche Qualità')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
            
        return cleaned_img


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
        total_files = len(os.listdir(f'{self.dataset_pth}{self.label_path}'))
        
        print(f"Inizio pulizia dataset - {total_files} file da processare")
        print(f"Cartella output: {clean_img_path}")
        
        for i, json_file in enumerate(os.listdir(f'{self.dataset_pth}{self.label_path}')):
            try:
                # Leggi il file JSON
                with open(f'{self.dataset_pth}{self.label_path}/{json_file}', 'r') as file:
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
                
                if doc_type == '03':
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
                # Salva l'immagine pulita
                output_filename = f"{filename}.jpg"
                output_filepath = os.path.join(output_path, output_filename)
                
                success = cv2.imwrite(output_filepath, cleaned_img)
                
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




    

                    