xwimport cv2
import numpy as np
from scipy.signal import find_peaks


class FindsLinesIMG:
    
    def __init__(self, low, high): 
        self.low_canny = low
        self.high_canny = high

    def give_tree_img(self, img):
        """
        Finds horizontal division points to split document image into 3 sections.
        
        This method analyzes the image structure to identify natural horizontal
        divisions that can be used to split a multi-document image into separate
        sections.
        
        The algorithm workflow:
        1. Convert input to numpy array if needed
        2. Apply Canny edge detection to find image edges
        3. Reinforce horizontal lines using morphological operations
        4. Create horizontal projection by summing pixel values along rows
        5. Find peaks in the inverted projection (valleys in original)
        6. Group nearby peaks together to identify stable division lines
        7. Return 4 y-coordinates defining 3 document sections
        
        Args:
            img: Input image (color or grayscale) to analyze for divisions
            
        Returns:
            tuple: Four y-coordinates (y0, y1, y2, y3) where:
                   - y0: Start of first section (always 0)
                   - y1: End of first section / Start of second section
                   - y2: End of second section / Start of third section  
                   - y3: End of third section (always image height)
        """
        # Ensure input is a numpy array for consistent processing
        if not isinstance(img, np.ndarray):
            img = np.array(img)

        # Step 1: Detect edges in the image using Canny algorithm
        edges = self._canny_edges(img)
        
        # Step 2: Reinforce horizontal lines with morphological operations
        # Kernel size is proportional to image width for adaptive processing
        kernel_size = max(15, img.shape[1] // 40)
        morphed = self._reinforce_horizontal_lines(edges, kernel_size)

        # Step 3: Create horizontal projection by summing along rows
        # This creates a 1D array where each value represents the total edge intensity in that row
        proj = np.sum(morphed, axis=1)
        
        # Step 4: Invert projection to find valleys (gaps between content)
        # Areas with less content will have higher values after inversion
        inv = np.max(proj) - proj
        
        # Step 5: Find peaks in inverted projection using scipy's find_peaks
        peaks, _ = find_peaks(inv,
                              height=np.mean(inv) * 0.03,      # Minimum peak height threshold
                              distance=50,                      # Minimum distance between peaks
                              prominence=np.std(inv) * 0.03)   # Minimum peak prominence

        # Step 6: Group nearby peaks to identify stable division lines
        sort_peaks = sorted(peaks)
        groups = []
        
        for y in sort_peaks:
            # If no groups exist or peak is far from last group, create new group
            if not groups or abs(y - groups[-1][-1]) > 5:
                groups.append([y])
            else:
                # Add peak to existing group if it's close enough (within 5 pixels)
                groups[-1].append(y)
        
        # Calculate average position for each group with multiple peaks
        # Only consider groups with multiple peaks as they are more reliable
        cut_ys = [int(np.mean(g)) for g in groups if len(g) > 1]

        # Step 7: Determine final division points based on found cuts
        h = img.shape[0]
        
        if len(cut_ys) >= 2:
            # Two or more reliable cuts found: use first two for 3 sections
            y0, y1, y2, y3 = 0, cut_ys[0], cut_ys[1], h
        elif len(cut_ys) == 1:
            # One reliable cut found: split into 2 sections, third is empty
            y0, y1, y2, y3 = 0, cut_ys[0], h, h
        else:
            # No reliable cuts found: divide into equal thirds as fallback
            y0, y1, y2, y3 = 0, h // 3, (2 * h) // 3, h

        return y0, y1, y2, y3

    # ==========================================
    # PRIVATE METHODS (Internal implementation)
    # ==========================================

    def _canny_edges(self, img_gray, low=None, high=None):
        """
        Applies Canny edge detection with automatic or manual threshold calculation.
        Args:
            img_gray: Input grayscale image
            low (int, optional): Lower threshold for edge detection. Uses class default if None
            high (int, optional): Upper threshold for edge detection. Uses class default if None
            
        Returns:
            numpy.ndarray: Binary edge image where edges are white (255) and background is black (0)
        """
        # Convert to grayscale if input is color image
        if img_gray.ndim == 3:
            img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)
        
        # Use provided thresholds or class defaults, with automatic calculation as fallback
        if low is None or high is None:
            if hasattr(self, 'low_canny') and hasattr(self, 'high_canny'):
                # Use class-defined thresholds
                low = self.low_canny
                high = self.high_canny
            else:
                # Calculate automatic thresholds based on image median
                v = np.median(img_gray)
                low = int(max(0, 0.66 * v))    # Lower threshold: 66% of median
                high = int(min(255, 1.33 * v)) # Upper threshold: 133% of median
        
        # Apply Canny edge detection with L2 gradient for better accuracy
        # L2gradient=True uses L2 norm for gradient magnitude calculation
        return cv2.Canny(img_gray, low, high, L2gradient=True)

    def _reinforce_horizontal_lines(self, edges, kernel_size=50):
        """
        Reinforces horizontal lines using morphological closing with horizontal kernel.
        Args:
            edges: Input binary edge image
            kernel_size (int): Length of horizontal kernel for morphological operation. Default: 50
            
        Returns:
            numpy.ndarray: Image with reinforced horizontal lines
        """
        # Ensure minimum kernel size and make it odd for proper morphological operations
        k = max(3, int(kernel_size))
        
        # Create horizontal rectangular kernel (width=k, height=1)
        # This kernel will connect horizontal elements while preserving vertical gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1))
        
        # Apply morphological closing to connect horizontal segments
        # Closing = dilation followed by erosion, fills small gaps in horizontal lines
        return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)