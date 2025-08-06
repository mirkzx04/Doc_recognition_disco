"""
CleanerIMG class for image data cleaning and preprocessing.

This module provides comprehensive image preprocessing functionality including
grayscale conversion, resizing, blurring, CLAHE enhancement, binarization,
morphological cleaning, and perspective correction.
"""

import numpy as np
import cv2
from scipy.signal import find_peaks


class CleanerIMG:
    """
    CleanerIMG class for comprehensive image data cleaning and preprocessing.
    
    This class provides methods for standardizing images through various preprocessing
    techniques including resizing, blurring, and enhancement operations.
    """

    def __init__(self, size=(224, 224), blur_kernel=(5, 5), min_area_ratio=0.01, 
                 aspect_range=(0.5, 3.0), margin=8, connect_ratio=120) -> None:
        """
        Initializes the CleanerIMG class with preprocessing parameters.

        Args:
            size (tuple): The target size for resizing images (width, height). Default: (224, 224)
            blur_kernel (tuple): The kernel size for Gaussian blur, must be odd and positive. Default: (5, 5)
            min_area_ratio (float): Minimum area ratio for contour filtering. Default: 0.01
            aspect_range (tuple): Valid aspect ratio range for documents (min, max). Default: (0.5, 3.0)
            margin (int): Margin to add around cropped regions. Default: 8
            connect_ratio (int): Controls kernel size for morphological operations. Default: 120

        Returns:
            None
        """
        # Store target image dimensions
        self.size = size
        
        # Store blur configuration
        self.blur_kernel = blur_kernel
        
        # Store document detection parameters
        self.min_area_ratio = min_area_ratio
        self.aspect_range = aspect_range
        self.margin = margin
        self.connect_ratio = connect_ratio

    def clahe_gray(self, img_bgr):
        """
        Converts BGR image to grayscale and applies CLAHE enhancement.
        
        CLAHE (Contrast Limited Adaptive Histogram Equalization) improves local contrast
        by applying histogram equalization in small regions called tiles.
        
        Args:
            img_bgr: Input BGR color image
            
        Returns:
            numpy.ndarray: Enhanced grayscale image
        """
        # Convert BGR to grayscale
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Create CLAHE object with specified parameters
        # clipLimit: threshold for contrast limiting
        # tileGridSize: size of the neighborhood area for histogram equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # Apply CLAHE to enhance local contrast
        return clahe.apply(gray)
    
    def binarize(self, img_gray):
        """
        Binarizes the grayscale image using Otsu's automatic thresholding.
        
        This method applies optional Gaussian blur followed by inverse Otsu thresholding
        to create a binary image where background is white (0) and content is black (255).
        
        Args:
            img_gray: Input grayscale image
            
        Returns:
            numpy.ndarray: Binary image with inverted threshold
        """
        # Apply Gaussian blur if kernel size is valid
        if len(self.blur_kernel) == 2 and self.blur_kernel[0] > 1:
            # Blur helps reduce noise before thresholding
            gray = cv2.GaussianBlur(img_gray, self.blur_kernel, 0)
        else:
            # Use original image if blur is disabled
            gray = img_gray

        # Apply inverse Otsu thresholding
        # THRESH_BINARY_INV: inverts the result (background white, content black)
        # THRESH_OTSU: automatically determines optimal threshold value
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return thresh
    
    def morph_clean(self, th):
        """
        Cleans binary image using morphological operations to remove noise.
        
        This method uses adaptive kernel sizing based on image dimensions to perform
        morphological opening followed by contour filtering based on area.
        
        Args:
            th: Input binary image
            
        Returns:
            numpy.ndarray: Cleaned binary image with noise removed
        """
        # Get image dimensions
        h, w = th.shape

        # Calculate adaptive kernel size based on image dimensions
        # Use bitwise OR with 1 to ensure odd kernel size
        kx = max(3, int(w // self.connect_ratio)) | 1
        ky = max(3, int(h // self.connect_ratio)) | 1

        # Create rectangular structuring element for morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
        
        # Apply morphological opening to remove small noise
        # Opening = erosion followed by dilation, removes small objects
        th2 = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)

        # Find all external contours in the processed image
        cnts, _ = cv2.findContours(th2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create empty mask for filtering components
        mask = np.zeros_like(th2)
        
        # Calculate total image area for ratio comparison
        total_area = w * h

        # Filter contours based on minimum area threshold
        for c in cnts:
            # Calculate contour area
            area = cv2.contourArea(c)
            
            # Keep only contours that meet minimum area requirement
            if area >= self.min_area_ratio * total_area:
                # Draw contour on mask with white fill
                cv2.drawContours(mask, [c], -1, 255, -1)

        return mask
    
    def resize_keep_ratio(self, img):
        """
        Resizes image while maintaining aspect ratio and adds white padding.
        
        This method scales the image to fit within the target size while preserving
        the original aspect ratio, then adds white padding to reach exact dimensions.
        
        Args:
            img: Input image to resize
            
        Returns:
            numpy.ndarray: Resized image with preserved aspect ratio and padding
        """
        # Get target and current image dimensions
        H, W = self.size
        h, w = img.shape[:2]

        # Calculate scale factor to fit image within target size
        # Use min to ensure image fits completely within target dimensions
        scale = min(W / max(1, w), H / max(1, h))
        
        # Calculate new dimensions after scaling
        nw, nh = int(round(w * scale)), int(round(h * scale))
        
        # Resize image using area interpolation (good for downscaling)
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

        # Calculate padding needed to reach target size
        dw, dh = W - nw, H - nh
        
        # Distribute padding evenly on both sides
        l, r = dw // 2, dw - dw // 2  # left, right padding
        t, b = dh // 2, dh - dh // 2  # top, bottom padding

        # Set border color: white for both grayscale and color images
        v = 255 if resized.ndim == 2 else (255, 255, 255)
        
        # Add constant white border to reach target dimensions
        return cv2.copyMakeBorder(resized, t, b, l, r, cv2.BORDER_CONSTANT, value=v)
    
    def warp_min_area_rect(self, img, cnt, margin=8, min_area=25.0, min_side=2):
        """
        Straightens and crops ROI defined by contour using minimum area rectangle.
        
        This method performs perspective correction on the region defined by a contour,
        with automatic fallback to bounding rectangle for degenerate/small contours.
        
        Args:
            img: Input image to warp
            cnt: Contour defining the region of interest
            margin (int): Margin to add around the cropped region. Default: 8
            min_area (float): Minimum contour area threshold. Default: 25.0
            min_side (int): Minimum side length for warped output. Default: 2
            
        Returns:
            numpy.ndarray: Warped and straightened image region
        """
        # Normalize contour format to expected shape
        if cnt.ndim == 2 and cnt.shape[1] == 2:
            # Convert (N, 2) to (N, 1, 2) format
            cnt_use = cnt.reshape(-1, 1, 2).astype(np.float32)
        elif cnt.ndim == 3 and cnt.shape[1] == 1 and cnt.shape[2] == 2:
            # Already in correct (N, 1, 2) format
            cnt_use = cnt.astype(np.float32)
        else:
            # Invalid contour format, return original image
            return img

        # Calculate contour area
        area = float(cv2.contourArea(cnt_use))
        
        # Use simple bounding rectangle for small contours
        if area < min_area:
            return self._fallback_bounding_rect(img, cnt_use, margin)

        # Get minimum area rectangle enclosing the contour
        rect = cv2.minAreaRect(cnt_use)
        
        # Get four corner points of the rectangle
        box = cv2.boxPoints(rect).astype(np.float32)  # Shape: (4, 2)

        # Sort points in clockwise order: top-left, top-right, bottom-right, bottom-left
        s = box.sum(axis=1)  # x + y for each point
        diff = np.diff(box, axis=1).ravel()  # x - y for each point
        
        # Find corners based on coordinate sums and differences
        tl = box[np.argmin(s)]      # smallest x + y (top-left)
        br = box[np.argmax(s)]      # largest x + y (bottom-right)
        tr = box[np.argmin(diff)]   # smallest x - y (top-right)
        bl = box[np.argmax(diff)]   # largest x - y (bottom-left)
        
        # Create ordered source points array
        src = np.stack([tl, tr, br, bl], axis=0).astype(np.float32)

        # Calculate output dimensions based on rectangle sides
        W = int(max(min_side, round(np.linalg.norm(br - bl))))  # width
        H = int(max(min_side, round(np.linalg.norm(tr - br))))  # height
        
        # Fallback to bounding rectangle if dimensions are too small
        if W < min_side or H < min_side:
            return self._fallback_bounding_rect(img, cnt_use, margin)

        # Define destination rectangle corners
        dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)

        try:
            # Calculate perspective transformation matrix
            M = cv2.getPerspectiveTransform(src, dst)
            
            # Apply perspective transformation
            warped = cv2.warpPerspective(img, M, (W, H), flags=cv2.INTER_CUBIC)
        except cv2.error:
            # Fallback to bounding rectangle if perspective transform fails
            return self._fallback_bounding_rect(img, cnt_use, margin)

        # Add margin around the warped image
        v = 255 if warped.ndim == 2 else (255, 255, 255)
        return cv2.copyMakeBorder(warped, margin, margin, margin, margin,
                                  cv2.BORDER_CONSTANT, value=v)

    # ==========================================
    # PRIVATE METHODS (Internal implementation)
    # ==========================================

    def _fallback_bounding_rect(self, img, cnt, margin):
        """
        Helper method that crops image using simple bounding rectangle.
        
        This is used as a fallback when perspective correction fails or
        when the contour is too small/degenerate for reliable warping.
        
        Args:
            img: Input image
            cnt: Contour in (N, 1, 2) format
            margin (int): Margin to add around the crop
            
        Returns:
            numpy.ndarray: Cropped image with margin
        """
        # Get axis-aligned bounding rectangle
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Calculate crop boundaries with margin, ensuring they stay within image bounds
        y0 = max(0, y - margin)
        y1 = min(img.shape[0], y + h + margin)
        x0 = max(0, x - margin)
        x1 = min(img.shape[1], x + w + margin)
        
        # Crop the image
        crop = img[y0:y1, x0:x1]
        
        # Add additional margin border
        v = 255 if crop.ndim == 2 else (255, 255, 255)
        return cv2.copyMakeBorder(crop, margin, margin, margin, margin, 
                                  cv2.BORDER_CONSTANT, value=v)