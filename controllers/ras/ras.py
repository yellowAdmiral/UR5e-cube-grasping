#FINALLL VERSION - MISSION 3
#Pranup Chhetri
import numpy as np
import kinpy as kp
import cv2
import imutils
from scipy.spatial.transform import Rotation as R
import math
from util import display_image, normalize_depth
from UR5e import UR5e

def detect_cube_centers(depth, min_depth, max_depth, image):
    """
    Detects the centers of cubes with red outlines in a depth image.

    Parameters:
    - depth: The input depth image (numpy array).
    - min_depth: Minimum depth value in millimeters to threshold the cubes.
    - max_depth: Maximum depth value in millimeters to threshold the cubes.
    - image: The input color image (numpy array).

    Returns:
    - centers_2d: List of 2D centers of the cubes.
    - angles: List of angles of the detected cubes.
    """
    # Apply a median filter to reduce noise
    depth_image = depth.copy()
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    depth_image_filtered = cv2.medianBlur(depth_image, 5)
    #Normalising image to [0, 255] range for 8-bit conversion
    depth_image_normalized = cv2.normalize(depth_image_filtered, None, 0, 255, cv2.NORM_MINMAX)

    # Convert the normalized image to 8-bit since contours needs 8-bit single-channel image (CV_8UC1)
    depth_image_8bit = np.uint8(depth_image_normalized)

    # Threshold depth image to isolate cubes
    _, thresholded = cv2.threshold(depth_image_8bit, min_depth, max_depth, cv2.THRESH_BINARY_INV)
    # Create a mask from the thresholded image
    mask = cv2.bitwise_not(thresholded)
    # display_image(thresholded, "Thresholded Image")
    #making the mask for red, using masks to detect the red outlines of the cubes
    #This is to make sure we select separate cubes when they are grouped together
    red_lower1 = np.array([0, 120, 70])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 120, 70])
    red_upper2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    
    #combining all the masks 
    final_mask = cv2.bitwise_and(red_mask, thresholded)
    # Apply the mask to the colour image and preparing for contour function
    final_img = cv2.bitwise_and(image, image, mask=final_mask)
    # display_image(final_img, "Masked + Thresholded Image")
    final_img_bw = cv2.cvtColor(final_img, cv2.COLOR_BGR2GRAY)
    final_img_bw = cv2.GaussianBlur(final_img_bw, (5, 5), 0)
    
    display_image(final_img_bw, "Final Masked Image")
    
    # Find contours and heirarchy 
    contours, hierarchy = cv2.findContours(final_img_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Initialize lists to hold the cube centers
    centers_2d = []
    angles = []

    # Iterate over contours to find and draw centers
    for i, contour in enumerate(contours):
        # If the contour has a parent (nested), check its size and proximity to the parent
        parent_idx = hierarchy[0][i][3]
        if parent_idx != -1:
            parent_contour = contours[parent_idx]
            parent_area = cv2.contourArea(parent_contour)
            area = cv2.contourArea(contour)
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            x, y = rect[0]
            width, height = rect[1]
            angle = rect[2]
            if area < 0.1 * parent_area:
                # Ignore small nested contours
                continue
            try:
                aspect_ratio = float(width) / height
                if aspect_ratio >0.7 and aspect_ratio < 1.2 and area < 500:
                    cv2.drawContours(depth_image, [box], -1, (0, 150, 0), 1)
                    if angle < -45:
                        angle += 90
                    angle_rad = np.deg2rad(angle)
                    # Calculate the center of the square
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        angles.append(angle_rad) 
                        centers_2d.append((cx, cy))
                        cv2.circle(depth_image, (cx, cy), 1, (150,120,0), -1)
            except ZeroDivisionError:
                pass
    display_image(depth_image, "Contours")
    return centers_2d, angles


def get_world_coordinates(pix_coord, camera_coord):
    """
    Function to convert the pixel coordinates to the world coordinates.
    It uses the formulae explained in the tutorial session.
    
    param pix_coord: tuple - (float, float), the (x,y) coordinates of the center of the cube face
    param camera_coord: tuple - (float, float, float), coordinates of the camera, but actually of the gripper
    
    returns:
        target: list of float, the calculated real world coordinates of the center of the cube
    """
    #Center of the input image and fov, already known.
    cx = 128//2
    cy = 64//2
    fov = 1
    #using the position of the end arm obtained using forward kinematics
    d = camera_coord[2] + 0.15 #0.15 is the z-axis offset of camera from the arm
    A = fov/2
    R = (d*(math.atan(A)))/cx #meters per pixel

    u, v = pix_coord

    dx = u - cx
    dy = v - cy
    x = R*dx
    y = R*dy
    offset = [ 0.0, 0.05, 0.0] #0.5 offset is the y-axis offset of the camera from the arm
    target = [camera_coord[0]+x+offset[0], camera_coord[1]-y+offset[1], 0.01+offset[2]]
    return target

    
def get_closest(objects):
    """
    Gets the closest cube to the robot's center.
    
    param objects: list, the list of pixel coordinates of the identified cubes
    
    returns:
    cur_min: int, the index of the closest object in objects list
    """
    distances = []
    cur_min = 0
    if len(objects)>0:
        cx = 128//2
        cy = 64//2
        for i in range(len(objects)):
            ox, oy = objects[i]
            dist = math.sqrt((ox - cx) ** 2 + (oy - cy) ** 2)
            distances.append(dist)
            if dist <= min(distances):
                cur_min = i
        return cur_min
    else:
        return 0
  
def main():
    """
    Main Function 
    Moves the robot arm to the home position then checks for cubes using depth
    and colour image. 
    If any cubes are detected, it translates the pixel coordinates to world 
    coordinates and tries to pick them up.
    Then it drops them into the yellow crate and repeats the above steps until
    no cubes are visible from home position.
    
    param: None
    returns: None
    """
    # initialise robot and move to home position
    robot = UR5e()
    robot.move_to_joint_pos(robot.home_pos)  # this is synchronised PTP with a timeout
    
    # use FK to get end-effector pose
    tf_ee = robot.forward_kinematics()  # check UR5e.py, this is preset to be the TCP (Tool Center Point)
    #Get depth and colour image
    img = robot.get_camera_image()
    image = img.copy()
    depth = robot.get_camera_depth_image()
    cam_pos = tf_ee.pos
    #look from visibile cubes using the function detect_cube_centers
    objects, angles = detect_cube_centers(depth, 250, 350, image)
    if len(objects) == 0:
        print("Nothing found, sorry!")
    while len(objects):
        #Look for closest object to make sure it doesnt go follows an order
        closest = get_closest(objects)
        object = objects[closest]
        angle = angles[closest]
        
        print("Found an object at ",object)
        cv2.circle(image, object, 1, (0,0,155), -1)
        display_image(image, "Current Target")
        target = get_world_coordinates(object, cam_pos)
        print("Picking cube at:", target)
        tf_target = kp.Transform(
                pos = target,
                rot = [-np.pi/2, 0, -angle]  # as rpy
            )
        # calculate IK
        joint_pos = robot.inverse_kinematics(tf_target)
        # move robot
        robot.move_to_joint_pos(joint_pos, velocity = 0.3, timeout = 10)
        robot.close_gripper()
    
        # move back to home configuration - note that we use very slow velocity
        robot.move_to_joint_pos(robot.home_pos, velocity=0.07, timeout = 30)
        
        end_target = kp.Transform(
                    pos = [-0.7, 0, 0.3],
                    rot = [-np.pi/2, 0, -np.pi/4]  # as rpy
                )
        joint_pos = robot.inverse_kinematics(end_target)
        robot.move_to_joint_pos(joint_pos, velocity=0.1, timeout = 30)
        robot.open_gripper()
        print("moving to home position")
        robot.move_to_joint_pos(robot.home_pos)
        img = robot.get_camera_image()
        image = img.copy()
        depth = robot.get_camera_depth_image()
        objects, angles = detect_cube_centers(depth, 250, 350, image)
        if len(objects) == 0:
            print("Thats all!!")

if __name__ == '__main__':
    main()
"""
Robot Architecture:
    The robot's architecture can be defined a simple state machine with
    deliberative and reactive controls. 
    States: 
        Initialization state: This can be defined as the starting state when
        the robot is moving to the home position.
        Working state: This is the state which starts after the robot tries
        detecting cubes and runs into the while loop.
    Behiaviour Control:
        The behaviour control of this robot comprises of both deliberative 
        and reactive control:
            Deliberative control: Actions like moving to home position before 
            detecting cubes and after dropping them off.
            Reactive Control: The detection of the cubes uses a reactive behaviour
            control mechanism.

Conclusion:
    The above implementation tries to ensure atleast one cube is piced up(ideally
    all cubes visible from the home position). The code exhibits the implementation
    of forward kinematics to find the current position of the robot arm and
    converting camera coordinates to world coordinates using the formula
    demonstrated in the tutorials. The cube detection is a combination of depth
    sensing and color masking, and finally, the dropping off of the cube is 
    a deliberative move to the middle of the yellow crate, about 30 cm above 
    it, using inverse kinematics.

Areas of improvement:
    The cube detection could be improved since it only checks for cubes from
    the home position. Additionally, the cubes are not allways correctly 
    identified, especially when they are clumped together.
    The angle detection could also be greatly improved since the cube slips
    out of the arm around half the time.
    Finally, there could be a way to figure out if the cube slipped out, maybe 
    by checking if there is somethging hanging on the arm using depth sensing.
        
 
 """