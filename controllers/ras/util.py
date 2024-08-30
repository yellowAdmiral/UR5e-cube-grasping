import numpy as np
import cv2


def display_image(image, name, scale=2, wait=False):
    """ 
    function to display an image 
    :param image: ndarray, the image to display
    :param name: string, a name for the window
    :param scale: int, optional, scaling factor for the image
    :param wait: bool, optional, if True, will wait for click/button to close window
    """
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, image.shape[1]*scale, image.shape[0]*scale)
    cv2.imshow(name, image)
    cv2.waitKey(0 if wait else 1)
    
    
def normalize_depth(depth_image):
    """
    function to normalize the depth image between 0 and 1 for better visualization.
    :param depth_image: 2d-array
    :returns: 2d-array, values between 0 and 1
    """
    if not isinstance(depth_image, np.ndarray):
        raise ValueError("Input must be a 2D NumPy array.")
        
    max_val = np.max(depth_image)
    
    if max_val == 0:
        return depth_image
    try :
        normalized_image = depth_image / max_val
    except :
        pass
    return normalized_image
