"""
StandardizationIMG class for image data standardization.

We convert images to grayscale, resize them to a specified size, adding blurring if needed.
"""

import numpy as np

import cv2

class StandardizationIMG:
    """
    StandardizationIMG class for image data standardization.
    """

    def __init__(self, size=(224, 224), blur = True, blur_kernel=(5, 5)) -> None:
        """
        Initializes the StandardizationIMG class.

        Args:
            size (tuple): The target size for resizing images (width, height).
            blur (bool): Whether to apply Gaussian blur to the images.
            blur_kernel (tuple): The kernel size for Gaussian blur, must be odd and positive.

        Returns:
            None
        """

        self.size = size

        self.blur = blur
        self.blur_kernel = blur_kernel
    
    def resize_keep_ratio(self, image: np.ndarray) -> np.ndarray:
        """
        Resizes the input image while maintaining its aspect ratio and adds padding to fit the target size.

        Args:
            image (numpy.ndarray): The input image to be resized.
        """

        if not isinstance(image, np.ndarray):
            image = np.ndarray(image)
        else:
            ValueError('Image is not numpy array')

        h, w = image.shape[:2]

        # Calculate the scale factor to maintain aspect ratio
        scale = min(self.size[0] / w, self.size[1] / h)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Adding padding to center image in to canvas
        delta_w = self.size[0] - new_w
        delta_h = self.size[1] - new_h
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        canvas = cv2.copyMakeBorder(
            resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value = 255
        )

        return canvas

    def standardize_image(self, image) -> np.ndarray:
        """
        Standardizes the input image by converting it to grayscale, resizing it

        Args:
            image (numpy.ndarray): The input image to be standardized.

        Returns:
            numpy.ndarray: The standardized image as a numpy array.
        
        Raises:
            TypeError: If the input image is not a numpy array.
        """

        if not isinstance(image, np.ndarray):
            raise TypeError("Input image must be a numpy array.")

        # Convert the image to grayscale and resize it
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized_image = self.resize_keep_ratio(gray_image)

        if self.blur:
            # Apply Gaussian blur to the resized image
            resized_image = cv2.GaussianBlur(resized_image, self.blur_kernel, 0)

        return np.expand_dims(resized_image, axis=-1)