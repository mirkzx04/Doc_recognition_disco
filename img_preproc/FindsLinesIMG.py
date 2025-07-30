import cv2
import numpy as np

class FindsLinesIMG:
    """
    Finds lines in an image using the Hough Transform.
    """

    def __init__(self, img, low, high, kernel_size=(5, 5)): 
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

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Finds lines in the input image using the Hough Transform.
        Args:
            img (numpy.ndarray): The input image in which to find lines. 
        Returns:
            numpy.ndarray: The image with detected lines drawn on it.
        """

        if not isinstance(img, np.ndarray):
            img = np.array(img)

        # Apply Canny edge detection to find edges in the image
        edges = cv2.Canny(img, low = self.low, high = self.high)

        #  Dilate the edges to enhance the line detection
        # This helps in connecting broken edges and making lines more prominent
        kernel = np.ones(self.kernel_size, np.uint8)
        edges = cv2.dilate(edges, kernel, iterations = 1)

        # Use the Hough Line Transform to detect lines in the edge-detected image
        lines = cv2.HoughLinesP(
            edges, 
            rho = 1, 
            theta = np.pi / 180, 
            threshold = 300, 
            minLineLength = 0.6 * img.shape[0], 
            maxLineGap = 20
        )

        candidate_lines = []
        for x1,y1,x2,y2 in lines.reshape(-1, 4):
            # Filter lines based on their length
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

            if abs(angle) < 5:
                candidate_lines.append((y1+y2)//2)

        candidate_lines.sort()
        groups = []
        for y in candidate_lines:
            if not groups or abs(y - groups[-1][-1]) > 5:
                groups.append([y])
            else:
                groups[-1].append(y)

        cut_ys = [int(np.mean(group)) for group in groups if len(group) > 1]

        y0, y1, y2, y3 = 0, cut_ys[0], cut_ys[1], img.shape[1]
        page1 = img[y0:y1]
        page2 = img[y1:y2]
        page3 = img[y2:y3]

        return page1, page2, page3
