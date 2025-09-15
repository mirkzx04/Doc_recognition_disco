import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

class FindsLinesIMG:
    """
    Classe per il rilevamento avanzato di linee nelle immagini di documenti.
    
    Utilizzata per:
    - Rilevamento bordi dei documenti
    - Analisi orientazione per correzione rotazione  
    - Segmentazione documenti multipli
    - Rilevamento strutture rettangolari
    """
    
    def __init__(self, low=50, high=150, debug=False):
        """
        Inizializza il rilevatore di linee.
        
        Args:
            low (int): Soglia bassa per Canny edge detection
            high (int): Soglia alta per Canny edge detection
            debug (bool): Attiva modalità debug
        """
        self.low_threshold = low
        self.high_threshold = high
        self.debug = debug
        
        # Parametri per Hough Line Detection
        self.hough_params = {
            'rho': 1,  # Risoluzione distanza in pixel
            'theta': np.pi/180,  # Risoluzione angolo in radianti
            'threshold': 100,  # Soglia accumulator
            'min_line_length': 50,  # Lunghezza minima linea
            'max_line_gap': 10  # Gap massimo tra segmenti
        }
        
        # Parametri per filtraggio linee
        self.line_params = {
            'angle_tolerance': 5.0,  # Tolleranza angolo in gradi
            'distance_tolerance': 20,  # Tolleranza distanza per raggruppamento
            'min_votes': 50  # Voti minimi per considerare una linea
        }
        
        if self.debug:
            print("🔍 FindsLinesIMG inizializzato")
    
    def detect_document_edges(self, img):
        """
        Rileva i bordi principali del documento.
        
        Args:
            img: Immagine in scala di grigi o colore
            
        Returns:
            dict: Dizionario con linee orizzontali e verticali principali
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
            
        # Edge detection
        edges = cv2.Canny(gray, self.low_threshold, self.high_threshold)
        
        # Rilevamento linee con Hough Transform
        lines = cv2.HoughLinesP(
            edges,
            self.hough_params['rho'],
            self.hough_params['theta'],
            self.hough_params['threshold'],
            minLineLength=self.hough_params['min_line_length'],
            maxLineGap=self.hough_params['max_line_gap']
        )
        
        if lines is None:
            return {'horizontal': [], 'vertical': [], 'all': []}
            
        # Classifica linee per orientazione
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calcola angolo della linea
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            angle = abs(angle)
            
            # Classifica come orizzontale o verticale
            if angle < self.line_params['angle_tolerance'] or angle > (180 - self.line_params['angle_tolerance']):
                horizontal_lines.append(line[0])
            elif abs(angle - 90) < self.line_params['angle_tolerance']:
                vertical_lines.append(line[0])
        
        # Raggruppa linee simili
        horizontal_groups = self._group_similar_lines(horizontal_lines, 'horizontal')
        vertical_groups = self._group_similar_lines(vertical_lines, 'vertical')
        
        if self.debug:
            self._visualize_lines(img, horizontal_groups, vertical_groups)
        
        return {
            'horizontal': horizontal_groups,
            'vertical': vertical_groups,
            'all': lines
        }
    
    def find_document_corners(self, img):
        """
        Trova gli angoli del documento usando intersezioni delle linee.
        
        Args:
            img: Immagine da analizzare
            
        Returns:
            np.array: Array di 4 punti rappresentanti gli angoli del documento
        """
        edge_data = self.detect_document_edges(img)
        
        if not edge_data['horizontal'] or not edge_data['vertical']:
            return None
            
        # Trova le linee più esterne
        h_lines = edge_data['horizontal']
        v_lines = edge_data['vertical']
        
        # Ordina linee orizzontali per coordinata y
        h_lines.sort(key=lambda line: (line[1] + line[3]) / 2)
        top_line = h_lines[0] if h_lines else None
        bottom_line = h_lines[-1] if h_lines else None
        
        # Ordina linee verticali per coordinata x
        v_lines.sort(key=lambda line: (line[0] + line[2]) / 2)
        left_line = v_lines[0] if v_lines else None
        right_line = v_lines[-1] if v_lines else None
        
        corners = []
        
        # Calcola intersezioni
        if top_line is not None and left_line is not None:
            intersection = self._line_intersection(top_line, left_line)
            if intersection:
                corners.append(intersection)
                
        if top_line is not None and right_line is not None:
            intersection = self._line_intersection(top_line, right_line)
            if intersection:
                corners.append(intersection)
                
        if bottom_line is not None and right_line is not None:
            intersection = self._line_intersection(bottom_line, right_line)
            if intersection:
                corners.append(intersection)
                
        if bottom_line is not None and left_line is not None:
            intersection = self._line_intersection(bottom_line, left_line)
            if intersection:
                corners.append(intersection)
        
        if len(corners) == 4:
            return np.array(corners, dtype=np.float32)
        else:
            return None
    
    def detect_skew_angle(self, img):
        """
        Rileva l'angolo di inclinazione del documento.
        
        Args:
            img: Immagine da analizzare
            
        Returns:
            float: Angolo di inclinazione in gradi
        """
        edge_data = self.detect_document_edges(img)
        
        if not edge_data['horizontal']:
            return 0.0
            
        # Analizza linee orizzontali per determinare inclinazione
        angles = []
        
        for line in edge_data['horizontal']:
            x1, y1, x2, y2 = line
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            angles.append(angle)
        
        if not angles:
            return 0.0
            
        # Usa la mediana per robustezza
        median_angle = np.median(angles)
        
        # Normalizza angolo nel range [-45, 45]
        if median_angle > 45:
            median_angle -= 90
        elif median_angle < -45:
            median_angle += 90
            
        return median_angle
    
    def segment_multi_documents(self, img):
        """
        Segmenta un'immagine contenente multipli documenti.
        
        Args:
            img: Immagine contenente multipli documenti
            
        Returns:
            list: Lista di bounding box dei documenti trovati
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
            
        # Edge detection aggressiva
        edges = cv2.Canny(gray, self.low_threshold//2, self.high_threshold)
        
        # Operazioni morfologiche per connettere bordi
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Trova contorni
        contours_result = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours_result[-2] if len(contours_result) >= 2 else contours_result[0]
        
        # Filtra contorni per dimensione e forma
        document_boxes = []
        img_area = img.shape[0] * img.shape[1]
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filtra per area (documenti devono essere significativi)
            if area < img_area * 0.05 or area > img_area * 0.8:
                continue
                
            # Calcola bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Verifica aspect ratio ragionevole per documenti
            aspect_ratio = w / h if h > 0 else 0
            if 0.3 < aspect_ratio < 4.0:
                document_boxes.append((x, y, w, h))
        
        # Ordina per area decrescente
        document_boxes.sort(key=lambda box: box[2] * box[3], reverse=True)
        
        if self.debug:
            self._visualize_document_boxes(img, document_boxes)
        
        return document_boxes[:3]  # Ritorna al massimo 3 documenti
    
    def _group_similar_lines(self, lines, orientation):
        """
        Raggruppa linee simili per orientazione e posizione.
        
        Args:
            lines: Lista di linee
            orientation: 'horizontal' o 'vertical'
            
        Returns:
            list: Lista di linee rappresentative per ogni gruppo
        """
        if not lines:
            return []
            
        # Converte linee in formato per clustering
        if orientation == 'horizontal':
            # Per linee orizzontali, usa coordinata y media
            features = [[(line[1] + line[3]) / 2] for line in lines]
        else:
            # Per linee verticali, usa coordinata x media
            features = [[(line[0] + line[2]) / 2] for line in lines]
        
        if len(features) < 2:
            return lines
            
        # Clustering DBSCAN per raggruppare linee vicine
        clustering = DBSCAN(eps=self.line_params['distance_tolerance'], min_samples=1)
        cluster_labels = clustering.fit_predict(features)
        
        # Trova linea rappresentativa per ogni cluster
        representative_lines = []
        for cluster_id in set(cluster_labels):
            if cluster_id == -1:  # Noise
                continue
                
            cluster_lines = [lines[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
            
            # Seleziona la linea più lunga del cluster
            longest_line = max(cluster_lines, key=lambda line: self._line_length(line))
            representative_lines.append(longest_line)
        
        return representative_lines
    
    def _line_length(self, line):
        """Calcola la lunghezza di una linea."""
        x1, y1, x2, y2 = line
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def _line_intersection(self, line1, line2):
        """
        Calcola l'intersezione tra due linee.
        
        Args:
            line1, line2: Linee nel formato [x1, y1, x2, y2]
            
        Returns:
            tuple: Coordinate del punto di intersezione o None
        """
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        
        # Calcola determinanti
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        
        if abs(denom) < 1e-10:  # Linee parallele
            return None
            
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
        
        # Calcola punto di intersezione
        intersection_x = x1 + t * (x2 - x1)
        intersection_y = y1 + t * (y2 - y1)
        
        return (int(intersection_x), int(intersection_y))
    
    def _visualize_lines(self, img, horizontal_lines, vertical_lines):
        """Visualizza le linee rilevate per debug."""
        if not self.debug:
            return
            
        vis_img = img.copy()
        if len(vis_img.shape) == 2:
            vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2BGR)
        
        # Disegna linee orizzontali in rosso
        for line in horizontal_lines:
            x1, y1, x2, y2 = line
            cv2.line(vis_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # Disegna linee verticali in blu
        for line in vertical_lines:
            x1, y1, x2, y2 = line
            cv2.line(vis_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
        plt.title('Linee Rilevate (Rosso=Orizzontali, Blu=Verticali)')
        plt.axis('off')
        plt.show()
    
    def _visualize_document_boxes(self, img, boxes):
        """Visualizza i bounding box dei documenti rilevati."""
        if not self.debug:
            return
            
        vis_img = img.copy()
        if len(vis_img.shape) == 2:
            vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2BGR)
        
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]  # Verde, Rosso, Blu
        
        for i, (x, y, w, h) in enumerate(boxes):
            color = colors[i % len(colors)]
            cv2.rectangle(vis_img, (x, y), (x + w, y + h), color, 3)
            cv2.putText(vis_img, f'Doc {i+1}', (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
        plt.title(f'Documenti Rilevati: {len(boxes)}')
        plt.axis('off')
        plt.show()
