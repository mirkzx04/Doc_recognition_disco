import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import feature, measure, filters, exposure, morphology
from sklearn.cluster import KMeans, DBSCAN
import warnings
warnings.filterwarnings('ignore')

class CleanerIMG:
    """
    Classe professionale per la pulizia avanzata di immagini di documenti.
    
    Gestisce:
    - Rimozione sfondi inutili con cropping intelligente
    - Correzione illuminazione non omogenea
    - Rilevamento e correzione rotazioni
    - Segmentazione documenti multipli
    - Valutazione qualità
    """
    
    def __init__(self, debug=False):
        """
        Inizializza il cleaner con parametri ottimizzati.
        
        Args:
            debug (bool): Se attivare modalità debug per visualizzazioni
        """
        self.debug = debug
        
        # Parametri per rilevamento contorni
        self.contour_params = {
            'min_area_ratio': 0.1,  # Minima area relativa del documento
            'max_area_ratio': 0.95,  # Massima area relativa del documento
            'aspect_ratio_tolerance': 0.1,  # Tolleranza per proporzioni
            'convexity_threshold': 0.85  # Soglia convessità per validazione forma
        }
        
        # Parametri per correzione illuminazione
        self.lighting_params = {
            'clahe_clip_limit': 2.0,
            'clahe_tile_size': (8, 8),
            'gamma_range': (0.7, 1.3),
            'shadow_threshold': 0.3
        }
        
        # Parametri per rilevamento rotazione
        self.rotation_params = {
            'hough_threshold': 100,
            'angle_tolerance': 2.0,  # gradi
            'min_line_length_ratio': 0.3,
            'max_line_gap_ratio': 0.1
        }
        
        if self.debug:
            print("🔧 CleanerIMG inizializzato in modalità DEBUG")
    
    def pipeline_clean(self, img, doc_type='01'):
        """
        Pipeline completa di pulizia adattiva per tipo documento.
        
        Args:
            img: Immagine da pulire
            doc_type: Tipo documento ('01'=tessera, '02'=foglio, '03'=multipli)
        
        Returns:
            Immagine pulita o None se pulizia fallisce
        """
        if img is None or img.size == 0:
            return None
            
        try:
            # Step 1: Preprocessing iniziale
            img_work = self._initial_preprocessing(img.copy())
            
            # Step 2: Rilevamento e correzione rotazione
            img_work = self._detect_and_correct_rotation(img_work, doc_type)
            
            # Step 3: Rimozione sfondo e cropping intelligente
            img_work = self._intelligent_crop(img_work, doc_type)
            
            # Step 4: Correzione illuminazione e ombre
            img_work = self._enhance_lighting(img_work)
            
            # Step 5: Pulizia finale
            img_work = self._final_cleanup(img_work, doc_type)
            
            if self.debug:
                self._show_pipeline_debug(img, img_work, doc_type)
                
            return img_work
            
        except Exception as e:
            import traceback
            if self.debug:
                print(f"⚠️ Errore durante pulizia: {e}")
                print(f"🔍 Traceback completo:")
                traceback.print_exc()
            return None
    
    def _initial_preprocessing(self, img):
        """Preprocessing iniziale: riduzione rumore e normalizzazione."""
        # Riduzione rumore
        img = cv2.bilateralFilter(img, 9, 75, 75)
        
        # Normalizzazione range colori
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        
        return img
    
    def _detect_and_correct_rotation(self, img, doc_type):
        """
        Rileva e corregge la rotazione dell'immagine usando multiple tecniche.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Metodo 1: Hough Lines per linee rette del documento
        angle_hough = self._detect_rotation_hough(gray)
        
        # Metodo 2: PCA dei contorni principali
        angle_pca = self._detect_rotation_pca(gray)
        
        # Metodo 3: Analisi gradiente per tessere
        if doc_type == '01':  # Tessere
            angle_gradient = self._detect_rotation_gradient(gray)
            angles = [angle_hough, angle_pca, angle_gradient]
        else:
            angles = [angle_hough, angle_pca]
        
        # Scegli l'angolo più affidabile
        angles = [a for a in angles if a is not None]
        if not angles:
            return img
            
        # Usa la mediana per robustezza
        final_angle = np.median(angles)
        
        # Correggi solo se rotazione significativa
        if abs(final_angle) > self.rotation_params['angle_tolerance']:
            img = self._rotate_image(img, final_angle)
            if self.debug:
                print(f"🔄 Rotazione corretta: {final_angle:.1f}°")
        
        return img
    
    def _detect_rotation_hough(self, gray):
        """Rileva rotazione usando Hough Line Transform."""
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=self.rotation_params['hough_threshold'])
        
        if lines is None:
            return None
            
        angles = []
        for line in lines[:20]:  # Analizza solo le prime 20 linee
            # Le linee di Hough possono avere formati diversi
            if isinstance(line, (list, tuple, np.ndarray)):
                if len(line) == 2:
                    rho, theta = line[0], line[1]
                elif len(line) == 1 and len(line[0]) >= 2:
                    rho, theta = line[0][0], line[0][1]
                else:
                    continue  # Skip linee malformate
            else:
                continue
                
            angle = theta * 180 / np.pi - 90
            if abs(angle) < 45:  # Considera solo angoli ragionevoli
                angles.append(angle)
        
        return np.median(angles) if angles else None
    
    def _detect_rotation_pca(self, gray):
        """Rileva rotazione usando PCA sui contorni principali."""
        # Trova contorni
        contours_result = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours_result[-2] if len(contours_result) >= 2 else contours_result[0]
        
        if not contours:
            return None
            
        # Seleziona il contorno più grande
        largest_contour = max(contours, key=cv2.contourArea)
        
        if len(largest_contour) < 10:
            return None
            
        # Calcola PCA
        points = largest_contour.reshape(-1, 2).astype(np.float32)
        mean, eigenvectors = cv2.PCACompute(points, mean=None)
        
        # Calcola angolo dal primo autovettore
        angle = np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0]) * 180 / np.pi
        
        return angle
    
    def _detect_rotation_gradient(self, gray):
        """Rileva rotazione usando analisi del gradiente (ottimo per tessere)."""
        # Calcola gradiente
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calcola orientazione gradiente
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        orientation = np.arctan2(grad_y, grad_x) * 180 / np.pi
        
        # Filtra per magnitudine significativa
        mask = magnitude > np.percentile(magnitude, 75)
        valid_orientations = orientation[mask]
        
        if len(valid_orientations) == 0:
            return None
            
        # Trova orientazione dominante
        hist, bins = np.histogram(valid_orientations, bins=180, range=(-90, 90))
        dominant_angle = bins[np.argmax(hist)]
        
        return dominant_angle
    
    def _rotate_image(self, img, angle):
        """Ruota l'immagine dell'angolo specificato."""
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        
        # Matrice di rotazione
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calcola nuove dimensioni per evitare crop
        cos_theta = np.abs(M[0, 0])
        sin_theta = np.abs(M[0, 1])
        new_w = int((h * sin_theta) + (w * cos_theta))
        new_h = int((h * cos_theta) + (w * sin_theta))
        
        # Aggiusta traslazione
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        
        # Applica rotazione
        rotated = cv2.warpAffine(img, M, (new_w, new_h), flags=cv2.INTER_CUBIC, 
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        
        return rotated
    
    def _intelligent_crop(self, img, doc_type):
        """
        Cropping intelligente per rimuovere sfondi inutili.
        Adattivo per tipo documento.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if doc_type == '01':  # Tessere - forma rettangolare ben definita
            return self._crop_card_document(img, gray)
        elif doc_type == '02':  # Fogli - forma rettangolare più grande
            return self._crop_sheet_document(img, gray)
        else:  # Documenti multipli - gestione speciale
            return self._crop_multi_document(img, gray)
    
    def _crop_card_document(self, img, gray):
        """Cropping specifico per documenti tipo tessera."""
        # Threshold adattivo per tessere
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        # Operazioni morfologiche per pulire
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Trova contorni
        contours_result = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours_result[-2] if len(contours_result) >= 2 else contours_result[0]
        
        if not contours:
            return img
            
        # Filtra contorni per tessere (aspect ratio circa 1.6:1)
        valid_contours = []
        img_area = img.shape[0] * img.shape[1]
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < img_area * self.contour_params['min_area_ratio']:
                continue
                
            # Calcola bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # Tessere hanno aspect ratio tra 1.4 e 1.8
            if 1.4 <= aspect_ratio <= 1.8:
                valid_contours.append((contour, area))
        
        if not valid_contours:
            return img
            
        # Seleziona il contorno più grande valido
        best_contour = max(valid_contours, key=lambda x: x[1])[0]
        
        # Ottieni bounding box con margine
        x, y, w, h = cv2.boundingRect(best_contour)
        margin = min(w, h) // 20  # Margine del 5%
        
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(img.shape[1], x + w + margin)
        y2 = min(img.shape[0], y + h + margin)
        
        return img[y1:y2, x1:x2]
    
    def _crop_sheet_document(self, img, gray):
        """Cropping specifico per documenti tipo foglio."""
        # Per fogli uso edge detection più aggressiva
        edges = cv2.Canny(gray, 30, 100)
        
        # Dilata gli edge per connettere bordi spezzati
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.dilate(edges, kernel, iterations=2)
        
        # Trova contorni
        contours_result = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours_result[-2] if len(contours_result) >= 2 else contours_result[0]
        
        if not contours:
            return img
            
        # Filtra per area e aspect ratio di fogli
        img_area = img.shape[0] * img.shape[1]
        valid_contours = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < img_area * 0.2:  # Fogli occupano più spazio
                continue
                
            # Approssima contorno
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Verifica se è simile a un rettangolo
            if len(approx) >= 4:
                valid_contours.append((contour, area))
        
        if not valid_contours:
            return img
            
        # Seleziona il contorno più grande
        best_contour = max(valid_contours, key=lambda x: x[1])[0]
        
        # Usa convex hull per ottenere forma più regolare
        hull = cv2.convexHull(best_contour)
        x, y, w, h = cv2.boundingRect(hull)
        
        # Margine più conservativo per fogli
        margin = min(w, h) // 50
        
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(img.shape[1], x + w + margin)
        y2 = min(img.shape[0], y + h + margin)
        
        return img[y1:y2, x1:x2]
    
    def _crop_multi_document(self, img, gray):
        """Cropping per documenti multipli - rimuove solo bordi ovvi."""
        # Per multipli, crop conservativo per non perdere documenti
        edges = cv2.Canny(gray, 50, 150)
        
        # Trova coordinate dei bordi attivi
        rows_with_content = np.where(np.sum(edges, axis=1) > 0)[0]
        cols_with_content = np.where(np.sum(edges, axis=0) > 0)[0]
        
        if len(rows_with_content) == 0 or len(cols_with_content) == 0:
            return img
            
        # Espandi leggermente i bordi
        margin = 20
        y1 = max(0, rows_with_content[0] - margin)
        y2 = min(img.shape[0], rows_with_content[-1] + margin)
        x1 = max(0, cols_with_content[0] - margin)
        x2 = min(img.shape[1], cols_with_content[-1] + margin)
        
        return img[y1:y2, x1:x2]
    
    def _enhance_lighting(self, img):
        """
        Migliora l'illuminazione correggendo non uniformità e ombre.
        """
        # Converte in LAB per lavorare sulla luminanza
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # CLAHE per migliorare contrasto locale
        clahe = cv2.createCLAHE(clipLimit=self.lighting_params['clahe_clip_limit'], 
                               tileGridSize=self.lighting_params['clahe_tile_size'])
        l_enhanced = clahe.apply(l)
        
        # Correzione illuminazione non uniforme
        l_corrected = self._correct_uneven_lighting(l_enhanced)
        
        # Rimuovi ombre
        l_no_shadows = self._remove_shadows(l_corrected)
        
        # Ricomponi immagine
        enhanced_lab = cv2.merge([l_no_shadows, a, b])
        enhanced_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # Gamma correction finale
        gamma = self._calculate_optimal_gamma(enhanced_img)
        enhanced_img = self._apply_gamma_correction(enhanced_img, gamma)
        
        return enhanced_img
    
    def _correct_uneven_lighting(self, l_channel):
        """Corregge illuminazione non uniforme."""
        # Stima dell'illuminazione di sfondo usando filtro gaussiano
        blurred = cv2.GaussianBlur(l_channel, (0, 0), sigmaX=l_channel.shape[1]/6)
        
        # Normalizza
        corrected = cv2.divide(l_channel, blurred, scale=128)
        
        return np.clip(corrected, 0, 255).astype(np.uint8)
    
    def _remove_shadows(self, l_channel):
        """Rimuove ombre usando morfologia."""
        # Rileva zone scure (potenziali ombre)
        mean_intensity = np.mean(l_channel)
        shadow_threshold = mean_intensity * self.lighting_params['shadow_threshold']
        
        shadow_mask = l_channel < shadow_threshold
        
        if np.sum(shadow_mask) == 0:
            return l_channel
            
        # Applica correzione selettiva alle zone d'ombra
        corrected = l_channel.copy()
        shadow_areas = l_channel[shadow_mask]
        
        # Alza la luminosità delle zone d'ombra
        brightness_boost = mean_intensity - np.mean(shadow_areas)
        corrected[shadow_mask] = np.clip(shadow_areas + brightness_boost * 0.7, 0, 255)
        
        return corrected.astype(np.uint8)
    
    def _calculate_optimal_gamma(self, img):
        """Calcola valore gamma ottimale per l'immagine."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Usa la media logaritmica per stimare gamma
        mean_log = np.mean(np.log(gray + 1))
        target_log = np.log(128)  # Target per immagine ben esposta
        
        gamma = target_log / mean_log if mean_log > 0 else 1.0
        
        # Clamp nel range ragionevole
        gamma = np.clip(gamma, 
                       self.lighting_params['gamma_range'][0], 
                       self.lighting_params['gamma_range'][1])
        
        return gamma
    
    def _apply_gamma_correction(self, img, gamma):
        """Applica correzione gamma."""
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                         for i in np.arange(0, 256)]).astype("uint8")
        
        return cv2.LUT(img, table)
    
    def _final_cleanup(self, img, doc_type):
        """Pulizia finale: sharpening e riduzione rumore."""
        # Unsharp masking per nitidezza
        blurred = cv2.GaussianBlur(img, (0, 0), 1.0)
        sharpened = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
        
        # Riduzione rumore finale
        cleaned = cv2.bilateralFilter(sharpened, 5, 50, 50)
        
        return cleaned
    
    def process_multi_documents(self, img, finder=None):
        """
        Gestisce documenti multipli separandoli in immagini individuali.
        
        Args:
            img: Immagine contenente multipli documenti
            finder: Istanza di FindsLinesIMG per rilevamento linee
            
        Returns:
            Lista di immagini dei documenti separati
        """
        if img is None:
            return []
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Metodo 1: Segmentazione per clustering colore
        documents_color = self._segment_by_color_clustering(img)
        
        # Metodo 2: Segmentazione per contorni
        documents_contour = self._segment_by_contours(img, gray)
        
        # Metodo 3: Segmentazione per watershed se disponibile
        documents_watershed = self._segment_by_watershed(img, gray)
        
        # Combina risultati dei diversi metodi
        all_documents = documents_color + documents_contour + documents_watershed
        
        # Filtra e valida i documenti estratti
        valid_documents = self._validate_extracted_documents(all_documents)
        
        # Ritorna al massimo 3 documenti (come richiesto)
        return valid_documents[:3]
    
    def _segment_by_color_clustering(self, img):
        """Segmentazione usando clustering dei colori."""
        # Reshape per clustering
        h, w, c = img.shape
        img_flat = img.reshape(-1, c)
        
        # K-means clustering per separare documenti da sfondo
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        labels = kmeans.fit_predict(img_flat)
        labels = labels.reshape(h, w)
        
        documents = []
        
        # Analizza ogni cluster
        for cluster_id in range(4):
            mask = (labels == cluster_id).astype(np.uint8) * 255
            
            # Trova contorni del cluster
            contours_result = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours_result[-2] if len(contours_result) >= 2 else contours_result[0]
            
            for contour in contours:
                area = cv2.contourArea(contour)
                img_area = h * w
                
                # Filtra per area ragionevole
                if 0.05 < area / img_area < 0.8:
                    x, y, w_cont, h_cont = cv2.boundingRect(contour)
                    
                    # Controlla aspect ratio ragionevole
                    aspect_ratio = w_cont / h_cont if h_cont > 0 else 0
                    if 0.5 < aspect_ratio < 3.0:
                        doc_crop = img[y:y+h_cont, x:x+w_cont]
                        if doc_crop.size > 0:
                            documents.append(doc_crop)
        
        return documents
    
    def _segment_by_contours(self, img, gray):
        """Segmentazione usando rilevamento contorni avanzato."""
        # Edge detection multi-scala
        edges1 = cv2.Canny(gray, 30, 100)
        edges2 = cv2.Canny(gray, 50, 150)
        edges = cv2.bitwise_or(edges1, edges2)
        
        # Operazioni morfologiche per connettere edge
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Trova contorni
        contours_result = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours_result[-2] if len(contours_result) >= 2 else contours_result[0]
        
        documents = []
        img_area = img.shape[0] * img.shape[1]
        
        # Ordina contorni per area
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        for contour in contours[:10]:  # Considera solo i 10 più grandi
            area = cv2.contourArea(contour)
            
            # Filtra per area
            if 0.1 < area / img_area < 0.9:
                # Approssima contorno
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Se ha forma rettangolare
                if len(approx) >= 4:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Verifica dimensioni ragionevoli
                    if w > 100 and h > 100:
                        doc_crop = img[y:y+h, x:x+w]
                        if doc_crop.size > 0:
                            documents.append(doc_crop)
        
        return documents
    
    def _segment_by_watershed(self, img, gray):
        """Segmentazione usando watershed algorithm."""
        # Threshold per separare foreground/background
        thresh_result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        thresh = thresh_result[1] if len(thresh_result) >= 2 else thresh_result[0]
        
        # Rimozione rumore
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Dilata per ottenere background sicuro
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # Distance transform per trovare foreground sicuro
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        fg_thresh_result = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        sure_fg = fg_thresh_result[1] if len(fg_thresh_result) >= 2 else fg_thresh_result[0]
        
        # Trova regione sconosciuta
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Labeling markers
        components_result = cv2.connectedComponents(sure_fg)
        markers = components_result[1] if len(components_result) >= 2 else components_result[0]
        markers = markers + 1
        markers[unknown == 255] = 0
        
        # Applica watershed
        img_for_watershed = img.copy()
        markers = cv2.watershed(img_for_watershed, markers)
        
        documents = []
        
        # Estrai ogni regione segmentata
        for label in np.unique(markers):
            if label <= 1:  # Skip background e bordi
                continue
                
            mask = np.zeros(gray.shape, dtype=np.uint8)
            mask[markers == label] = 255
            
            # Trova bounding box della regione
            contours_result = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours_result[-2] if len(contours_result) >= 2 else contours_result[0]

            if contours:
                x, y, w, h = cv2.boundingRect(contours[0])
                
                # Filtra per dimensioni
                if w > 100 and h > 100:
                    doc_crop = img[y:y+h, x:x+w]
                    if doc_crop.size > 0:
                        documents.append(doc_crop)
        
        return documents
    
    def _validate_extracted_documents(self, documents):
        """Valida e filtra i documenti estratti."""
        if not documents:
            return []
            
        valid_docs = []
        
        for doc in documents:
            # Calcola metriche di qualità
            quality = self.assess_quality(doc)
            
            # Filtra per qualità minima
            if quality['overall'] > 0.2:
                # Controlla se non è duplicato
                is_duplicate = False
                for existing_doc in valid_docs:
                    if self._are_documents_similar(doc, existing_doc):
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    valid_docs.append(doc)
        
        # Ordina per qualità decrescente
        valid_docs.sort(key=lambda x: self.assess_quality(x)['overall'], reverse=True)
        
        return valid_docs
    
    def _are_documents_similar(self, doc1, doc2, threshold=0.8):
        """Verifica se due documenti sono simili (per evitare duplicati)."""
        try:
            # Ridimensiona per confronto veloce
            doc1_small = cv2.resize(doc1, (100, 100))
            doc2_small = cv2.resize(doc2, (100, 100))
            
            # Calcola correlazione
            corr = cv2.matchTemplate(doc1_small, doc2_small, cv2.TM_CCOEFF_NORMED)
            
            return np.max(corr) > threshold
            
        except:
            return False
    
    def assess_quality(self, img):
        """
        Valuta la qualità dell'immagine con metriche multiple.
        
        Returns:
            dict: Dizionario con metriche di qualità
        """
        if img is None or img.size == 0:
            return {'overall': 0.0, 'sharpness': 0.0, 'contrast': 0.0, 'brightness': 0.0}
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Sharpness usando varianza del Laplaciano
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_norm = min(sharpness / 1000.0, 1.0)  # Normalizza
        
        # Contrast usando standard deviation
        contrast = np.std(gray) / 128.0  # Normalizza a [0,1]
        contrast_norm = min(contrast, 1.0)
        
        # Brightness - preferisce valori attorno a 128
        brightness = np.mean(gray)
        brightness_norm = 1.0 - abs(brightness - 128) / 128.0
        
        # Punteggio overall pesato
        overall = (sharpness_norm * 0.4 + contrast_norm * 0.4 + brightness_norm * 0.2)
        
        return {
            'overall': overall,
            'sharpness': sharpness_norm,
            'contrast': contrast_norm,
            'brightness': brightness_norm
        }
    
    def _show_pipeline_debug(self, original, processed, doc_type):
        """Mostra debug della pipeline."""
        if not self.debug:
            return
            
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        plt.title(f'Originale (Tipo: {doc_type})')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
        plt.title('Processata')
        plt.axis('off')
        
        # Mostra metriche
        quality = self.assess_quality(processed)
        plt.subplot(1, 3, 3)
        metrics = list(quality.keys())
        values = list(quality.values())
        plt.bar(metrics, values)
        plt.title('Metriche Qualità')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
