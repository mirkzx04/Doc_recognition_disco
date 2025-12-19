import cv2
import numpy as np

from rembg import remove, new_session

class DynamicDocumentCleaner:
    def __init__(self, debug=False):
        """
        Initialize the document cleaning class.
        
        :param debug: If True, shows intermediate steps (useful for tuning).
        """
        self.debug = debug
        # self.session = new_session('isnet-general-use')

    def order_points(self, pts):
        """
        Order the coordinates of the 4 document corners:
        [top-left, top-right, bottom-right, bottom-left].
        Required for perspective transformation.
        """
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # Top-left has the smallest sum
        rect[2] = pts[np.argmax(s)]  # Bottom-right has the largest sum

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # Top-right has the smallest difference
        rect[3] = pts[np.argmax(diff)]  # Bottom-left has the largest difference
        return rect

    def adaptive_light_correction(self, image):
        """
        Correct overexposed or underexposed areas by converting to LAB color space
        and applying CLAHE only to the luminance (L) channel.
        """
        # Convert to LAB (Luminance, A-channel, B-channel)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # High clipLimit corrects strong contrast, tileGridSize defines locality
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)

        # Merge channels and convert back to BGR
        limg = cv2.merge((cl, a, b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return final

    def remove_noise(self, image):
        """
        Apply Bilateral Filter to remove noise while keeping text edges sharp.
        """
        # d: diameter of the pixel neighborhood
        # sigmaColor: how similar colors must be to be blended
        # sigmaSpace: how spatially close pixels must be
        return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

    def get_document_mask(self, image):
        """
        Use U-2-Net to create a image mask
        """
        output = remove(image) # remove background
        mask = output[:, :, 3]

        # Clean the mask
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

        return mask
    
    def find_contour(self, mask):
        """
        Find the contour 
        """
        contours , _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None
        
        largest_contour = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)

        if len(approx) == 4:
            return approx
        else : 
            rect = cv2.minAreaRect(largest_contour)
            box = cv2.boxPoints(rect)
            box = box.astype(np.int32)

            return box

    def four_point_transform(self, image, pts):
        """
        Perform perspective warp to obtain a top-down view.
        """
        rect = self.order_points(pts.reshape(4, 2))
        (tl, tr, br, bl) = rect

        # Calculate maximum width
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        # Calculate maximum height
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        # Destination points for the flat view
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        # Calculate transformation matrix and apply
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped

    def process(self, img_matrix, image_path):
        """
        Main processing pipeline.

        Args: 
            img_matrix (np.array) : numpy array of original image
            image_path (str) : String of image path
        """
        # 1. Load the image
        original_image = cv2.cvtColor(img_matrix, cv2.COLOR_RGB2BGR)
        if original_image is None:
            raise ValueError(f"Unable to load image {image_path}")

        # 2. Light correction (pre-processing to help detection)
        # Note: We use a copy for detection, but apply crop on the original
        # or enhanced original to maintain quality.
        enhanced = self.adaptive_light_correction(original_image)
        denoised = self.remove_noise(enhanced)
        mask = self.get_document_mask(denoised)
        contour = self.find_contour(mask)

        if contour is not None:
            warped = self.four_point_transform(original_image, contour)
            return warped
        else : 
            return denoised