import cv2
import numpy as np

from scipy.signal import find_peaks

class FindsLinesIMG:
    """
    Finds lines in an image using the Hough Transform.
    """

    def __init__(self, low, high): 
        """
        Initializes the FindsLinesIMG class, idenify the 3 documents in the comsposite image.

        Args:
            low (int): The lower threshold for the Canny edge detector.
            high (int): The upper threshold for the Canny edge detector.
        Returns:
            None
        """

        self.low_canny = low
        self.high_canny = high

    def find_horizontal_lines(self, img: np.ndarray) -> np.ndarray:
        """
        Finds lines in the input image using the Hough Transform.
        Args:
            img (numpy.ndarray): The input image in which to find lines. 
        Returns:
            numpy.ndarray: The image with detected lines drawn on it.
        """

        # Horizontal projection (sum each lines)
        horizontal_projection = np.sum(img, axis=1)

        # Find row with low content, reverse the horizontal projection to find the lows as the peaks
        inverted = np.max(horizontal_projection) - horizontal_projection
        peaks, _ = find_peaks(inverted, height=np.mean(inverted) * 0.03, 
                              distance=50, prominence=np.std(inverted) * 0.03)

        return peaks
    
    def crop_document(self, image):
        """
        Crops the image to the largest external contour found.
        Args:
            image (numpy.ndarray): Grayscale input image.
        Returns:
            numpy.ndarray: Cropped image or original if no contours found.
        """

        thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 15, 8)
        
        # Find borders
        borders, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not borders:
            return image
        
        # Taked the bigger borders
        x, y, w, h = cv2.boundingRect(max(borders, key=cv2.contourArea))
        cropped = image[y:y+h, x:x+w]

        return cropped
    
    def reinforce_hirozontal_lines(self, edges, kernel_size = 50):
        """
        Reinforces horizontal lines in a binary edge image using morphological closing.
        This function applies a horizontal rectangular structuring element to the input edge image,
        enhancing and connecting horizontal lines by performing a morphological closing operation.
        Args:
            edges (numpy.ndarray): Binary edge image (single-channel) where horizontal lines are to be reinforced.
            kernel_size (int, optional): Length of the horizontal structuring element. Default is 50.
        Returns:
            numpy.ndarray: Image with reinforced horizontal lines.
        """

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(kernel_size), 1))
        morphed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        return morphed
    
    def extract_edges(self, image):
        return cv2.Canny(image, self.low_canny, self.high_canny)
    
    def give_tree_img(self, image):
        """
        Splits a blurred image into three horizontal sections based on detected horizontal lines.
        This method processes the input image to detect horizontal lines using Canny edge detection
        and a custom line grouping algorithm. It then determines the most probable cut positions
        (y-coordinates) for splitting the image into three parts. If no or only one line is detected,
        the image is divided into three equal or two unequal parts, respectively.
        Args:
            image_blur (numpy.ndarray or array-like): The blurred grayscale image to be split.
        Returns:
            tuple: A tuple containing three numpy.ndarray objects, each representing a horizontal section
                   of the original image (page_1, page_2, page_3).
        """

        if not isinstance(image, np.ndarray):
            image = np.array(image)

        edges = self.extract_edges(image)
        morphed = self.reinforce_hirozontal_lines(edges)
        peaks = self.find_horizontal_lines(morphed)

        sort_peaks = sorted(peaks)
        groups = []

        for y in sort_peaks:
            if not groups or abs(y - groups[-1][-1]) > 5:
                groups.append([y])
            else:
                groups[-1].append(y)
        
        cut_ys = [int(np.mean(group)) for group in groups if len(group) > 1]

        if len(cut_ys) >= 2:
            y0, y1, y2, y3 = 0, cut_ys[0], cut_ys[1], image.shape[0]
        elif len(cut_ys) == 1:
            h = image.shape[0]
            y0, y1, y2, y3 = 0, cut_ys[0], h, h
        else:
            # Divide into trhee equal parts
            h = image.shape[0]
            y0, y1, y2, y3 = 0, h//3, h*2//3, h

        # Cut image
        page_1 = image[y0:y1]
        page_2 =image[y1:y2]
        page_3 = image[y2:y3]

        return y0, y1, y2, y3



