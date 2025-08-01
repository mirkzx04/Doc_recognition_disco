import cv2
import numpy as np

from scipy.signal import find_peaks

from .StandardizationIMG import StandardizationIMG

class FindsLinesIMG:
    """
    Finds lines in an image using the Hough Transform.
    """

    def __init__(self, low, high, kernel_size=(5, 5)): 
        """
        Initializes the FindsLinesIMG class, idenify the 3 documents in the comsposite image.

        Args:
            img (numpy.ndarray): The input image in which to find lines.
            low (int): The lower threshold for the Canny edge detector.
            high (int): The upper threshold for the Canny edge detector.
        Returns:
            None
        """

        self.low_canny = low
        self.high_canny = high

        self.kernel_size = kernel_size

        self.standardizer = StandardizationIMG()

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
        peaks, _ = find_peaks(inverted, height=np.mean(inverted) * 0.3, distance=10, prominence=np.std(inverted) * 0.3)

        return peaks
    
    def give_tree_img(self, image_blur):
        if not isinstance(image_blur, np.ndarray):
            image_blur = np.array(image_blur)

        edges = cv2.Canny(image_blur, self.low_canny, self.high_canny)
        # Use find_horizontal_lines
        candidate_lines = self.find_horizontal_lines(edges)
        candidate_lines = candidate_lines.tolist()

        print(f'Number of candidate lines : {len(candidate_lines)}')

        candidate_lines.sort()
        groups = []

        for y in candidate_lines:
            if not groups or abs(y - groups[-1][-1]) > 5:
                groups.append([y])
            else:
                groups[-1].append(y)
        
        cut_ys = [int(np.mean(group)) for group in groups if len(group) > 1]

        if len(cut_ys) >= 2:
            y0, y1, y2, y3 = 0, cut_ys[0], cut_ys[1], image_blur.shape[0]
        elif len(cut_ys) == 1:
            h = image_blur.shape[0]
            y0, y1, y2, y3 = 0, cut_ys[0], h, h
        else:
            # Divide into trhee equal parts
            h = image_blur.shape[0]
            y0, y1, y2, y3 = 0, h//3, h*3//3, h

        # Cut image
        page_1 = image_blur[y0:y1]
        page_2 =image_blur[y1:y2]
        page_3 = image_blur[y2:y3]

        return y0, y1, y2, y3



