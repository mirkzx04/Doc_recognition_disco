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
    
    def __init__(self, debug=False, conservative_crop=True):
        """
        Inizializza il cleaner con parametri ottimizzati.
        
        Args:
            debug (bool): Se attivare modalità debug per visualizzazioni
            conservative_crop (bool): Se usare cropping conservativo
        """
        self.debug = debug
        self.conservative_crop = conservative_crop
        
        # Parametri per rilevamento contorni - AGGIORNATI
        self.contour_params = {
            'min_area_ratio': 0.3 if conservative_crop else 0.1,  # Aumentato
            'max_area_ratio': 0.95,
            'aspect_ratio_tolerance': 0.2 if conservative_crop else 0.1,  # Più permissivo
            'convexity_threshold': 0.75 if conservative_crop else 0.85,  # Ridotto per forme meno perfette
            'border_proximity_weight': 2.0  # Nuovo: peso per vicinanza ai bordi
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
        
        self.light_target_mean = 0.52   # target luminosità (0..1)
        self.min_doc_area_card = 0.30 if conservative_crop else 0.20
        self.min_doc_area_sheet = 0.45 if conservative_crop else 0.30

        if self.debug:
            print("🔧 CleanerIMG inizializzato in modalità DEBUG")
    
    # === Migliorie chiave ===
    def _white_balance_grayworld(self, img):
        b, g, r = cv2.split(img.astype(np.float32))
        mean_b, mean_g, mean_r = [np.mean(c) + 1e-6 for c in (b, g, r)]
        k = (mean_r + mean_g + mean_b) / 3.0
        b *= k / mean_b; g *= k / mean_g; r *= k / mean_r
        wb = np.clip(cv2.merge([b, g, r]), 0, 255).astype(np.uint8)
        return wb

    def _normalize_lighting(self, img):
        img = self._white_balance_grayworld(img)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=self.lighting_params['clahe_clip_limit'],
            tileGridSize=self.lighting_params['clahe_tile_size']
        )
        Lc = clahe.apply(L)
        lab = cv2.merge([Lc, A, B])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=25, sigmaY=25)
        bg = np.maximum(bg, 1.0)
        norm = (gray / bg)
        norm = np.clip(norm, 0.25, 2.0)  # clamp per evitare banding/zone nere
        norm = cv2.normalize(norm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # riallinea la luminanza al target
        m = norm.mean() / 255.0
        gain = np.clip(self.light_target_mean / max(m, 1e-3), 0.7, 1.4)
        L_adj = np.clip(norm.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(cv2.merge([L_adj, A, B]), cv2.COLOR_LAB2BGR)
        return img

    # ---------- Rilevamento documento ----------
    def _find_document_mask(self, gray):
        # threshold adattivo + morfologia per isolare il foglio/card
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        thr = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 31, 5)
        # inverti se serve per avere documento=bianco
        if np.mean(thr[:10, :]) + np.mean(thr[-10:, :]) + np.mean(thr[:, :10]) + np.mean(thr[:, -10:]) < 4*127:
            thr = cv2.bitwise_not(thr)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
        mask = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.medianBlur(mask, 5)

        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[-2] if len(cnts) >= 2 else cnts[0]
        if not cnts:
            return None, None

        H, W = gray.shape[:2]
        img_area = H * W
        best = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(best)
        return best, area / img_area

    def _crop_from_contour(self, img, cnt):
        rect = cv2.minAreaRect(cnt)
        return self._crop_min_area_rect(img, rect)

    def _largest_doc_rect(self, gray, min_area_ratio=0.25, max_area_ratio=0.98):
        # versione precedente (fallback)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        edges = cv2.Canny(blur, 80, 200)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,(9,9)))
        cnts = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[-2] if len(cnts) >= 2 else cnts[0]
        if not cnts:
            return None
        H, W = gray.shape[:2]; img_area = H * W
        best = None; best_score = -1.0
        for c in cnts:
            area = cv2.contourArea(c)
            if area < img_area * min_area_ratio or area > img_area * max_area_ratio:
                continue
            rect = cv2.minAreaRect(c)
            (w, h) = rect[1]
            if w < 5 or h < 5: 
                continue
            box = cv2.boxPoints(rect).astype(np.int32)
            x, y, ww, hh = cv2.boundingRect(box)
            border_touch = int(x < 0.05*W) + int(y < 0.05*H) + int(x+ww > 0.95*W) + int(y+hh > 0.95*H)
            rectangularity = area / float(ww*hh + 1e-6)
            score = rectangularity + 0.25*border_touch + (area/img_area)
            if score > best_score:
                best_score = score; best = rect
        return best

    def _crop_min_area_rect(self, img, rect, margin_ratio=0.01):
        box = cv2.boxPoints(rect).astype(np.float32)
        w, h = rect[1]
        if w < 1 or h < 1:
            return img
        s = box.sum(axis=1); d = np.diff(box, axis=1).ravel()
        tl = box[np.argmin(s)]; br = box[np.argmax(s)]
        tr = box[np.argmax(d)];  bl = box[np.argmin(d)]
        src = np.array([tl,tr,br,bl], dtype=np.float32)
        target_w = int(round(max(w, h))); target_h = int(round(min(w, h))) if w > h else int(round(max(w, h)))
        if target_w < 20 or target_h < 20:
            return img
        dst = np.array([[0,0],[target_w-1,0],[target_w-1,target_h-1],[0,target_h-1]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(img, M, (target_w, target_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        m = int(margin_ratio * min(target_w, target_h))
        if m > 0 and target_w > 2*m and target_h > 2*m:
            warped = warped[m:target_h-m, m:target_w-m]
        return warped

    def _deskew_sheet(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray,(3,3),0), 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 150)
        if lines is None:
            return img
        angles = []
        for rho, theta in lines[:,0]:
            ang = (theta * 180/np.pi) - 90
            if -45 <= ang <= 45:
                angles.append(ang)
        if not angles:
            return img
        angle = np.median(angles)
        if abs(angle) < 5:
            return img
        (h, w) = img.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    def _split_card_front_back(self, img):
        # se due card impilate (fronte/retro), prova split orizzontale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        proj = np.mean(gray, axis=1)
        # smussa e trova valle importante vicino al centro
        sm = cv2.GaussianBlur(proj.reshape(-1,1), (0,0), 7).ravel()
        H = len(sm)
        mid = H//2
        window = slice(int(H*0.3), int(H*0.7))
        idx = int(np.argmin(sm[window])) + int(H*0.3)
        # controlla che siano due blocchi separati (bordo chiaro nel mezzo)
        if abs(idx - mid) < int(H*0.15):
            top_mean = np.mean(sm[:idx]); bot_mean = np.mean(sm[idx:])
            if abs(top_mean - bot_mean) < 10:  # differenza contenuta
                return [img]  # non split
        # verifica altezze minime
        if idx > int(H*0.25) and (H-idx) > int(H*0.25):
            return [img[:idx, :], img[idx:, :]]
        return [img]

    # ---------- Pipeline principale ----------
    def pipeline_clean(self, img, doc_type):
        img = self._normalize_lighting(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        cnt, area_ratio = self._find_document_mask(gray)
        if cnt is not None:
            img_crop = self._crop_from_contour(img, cnt)
        else:
            rect = self._largest_doc_rect(gray, min_area_ratio=0.25, max_area_ratio=0.98)
            img_crop = self._crop_min_area_rect(img, rect) if rect is not None else img

        # Gate sull’area: se troppo piccolo, usa fallback più permissivo
        H0, W0 = img.shape[:2]
        Hc, Wc = img_crop.shape[:2]
        area_ok = (Hc*Wc) >= (H0*W0*(self.min_doc_area_card if doc_type != '02' else self.min_doc_area_sheet))
        if not area_ok:
            rect2 = self._largest_doc_rect(gray, min_area_ratio=0.15, max_area_ratio=0.995)
            if rect2 is not None:
                img_crop = self._crop_min_area_rect(img, rect2)

        if doc_type == '02':
            img_crop = self._deskew_sheet(img_crop)

        # Se è una card, prova a splittare fronte/retro
        if doc_type != '02':
            parts = self._split_card_front_back(img_crop)
            # per ora ritorna la parte più grande (puoi salvare entrambe nel multi-doc pipeline)
            img_crop = max(parts, key=lambda im: im.shape[0]*im.shape[1])

        return img_crop
    
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
