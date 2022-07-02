#!/usr/bin/env python3
"""
'detect_py35.py'

Detects a target by using color range segmentation with OpenCV.

"""

import cv2

# Constants
PERCENT_CENTER_RECT = 0.20  # For calculating the center rectangle's size
PERCENT_TARGET_RADIUS = 0.25 * PERCENT_CENTER_RECT  # Minimum target radius to follow
NUM_FILT_POINTS = 20  # Number of filtering points for the Moving Average Filter
DESIRED_IMAGE_HEIGHT = 480  # A smaller image makes the detection less CPU intensive

# A dictionary of two empty buffers (arrays) for the Moving Average Filter
filt_buffer = {'width': [], 'height': []}

# A dictionary of general parameters derived from the camera image size,
# which will be populated later with the 'get_cam_params' function
params = {'image_height': None, 'image_width': None, 'resized_height': None, 'resized_width': None,
          'x_axe_pos': None, 'y_axe_pos': None, 'cent_rect_half_width': None, 'cent_rect_half_height': None,
          'cent_rect_p1': None, 'cent_rect_p2': None, 'scaling_factor': None, 'min_tgt_radius': None}


# ---------- This is the application's 'main' asynchronousq function ----------
def run():
    """ Detects a target by using color range segmentation. """

    # Open the video camera
    vid_cam = cv2.VideoCapture(0)

    # Check if the camera opened correctly
    if vid_cam.isOpened() is False:
        print('[ERROR] Couldnt open the camera.')
        return

    print('-- Camera opened successfully')

    # Compute general parameters
    get_cam_params(vid_cam)
    # print(f"-- Original image width, height: {params['image_width']}, {params['image_height']}")

    # Infinite detect-follow loop
    while True:
        # Get the target coordinates (if any target was detected)
        tgt_cam_coord, frame, contour = get_target_coordinates(vid_cam)

        # print('-- break point 1')
        # If a target was found, filter their coordinatesq
        if tgt_cam_coord['width'] is not None and tgt_cam_coord['height'] is not None:
            # Apply Moving Average filter to target camera coordinates
            tgt_filt_cam_coord = moving_average_filter(tgt_cam_coord)

        # No target was found, set target camera coordinates to the Cartesian origin,
        # so the drone doesn't move
        else:
            # The Cartesian origin is where the x and y Cartesian axes are located
            # in the image, in pixel units
            tgt_cam_coord = {'width': params['y_axe_pos'],
                             'height': params['x_axe_pos']}  # Needed just for drawing objects
            tgt_filt_cam_coord = {'width': params['y_axe_pos'], 'height': params['x_axe_pos']}

        # Convert from camera coordinates to Cartesian coordinates (in pixel units)
        tgt_cart_coord = {'x': (tgt_filt_cam_coord['width'] - params['y_axe_pos']),
                          'y': (params['x_axe_pos'] - tgt_filt_cam_coord['height'])}

        # Compute scaling conversion factor from camera coordinates in pixel units
        # to Cartesian coordinates in meters
        COORD_SYS_CONV_FACTOR = 0.1

        # If the target is outside the center rectangle, compute North and East coordinates 
        if abs(tgt_cart_coord['x']) > params['cent_rect_half_width'] or \
                abs(tgt_cart_coord['y']) > params['cent_rect_half_height']:
            # Compute North, East coordinates applying "camera pixel" to Cartesian conversion factor
            E_coord = tgt_cart_coord['x'] * COORD_SYS_CONV_FACTOR
            N_coord = tgt_cart_coord['y'] * COORD_SYS_CONV_FACTOR
            # D_coord, yaw_angle don't change

        # Draw objects over the detection image frame just for visualization
        frame = draw_objects(tgt_cam_coord, tgt_filt_cam_coord, frame, contour)

        # Show the detection image frame on screen
        cv2.imshow("Detection of fire", frame)

        # Catch aborting key from computer keyboard
        key = cv2.waitKey(1) & 0xFF
        # If the 'q' key is pressed, break the 'while' infinite loop
        if key == ord("q"):
            break

    print("The script has ended.")


def get_cam_params(vid_cam):
    """ Computes useful general parameters derived from the camera image size."""

    # Grab a frame and get its size
    is_grabbed, frame = vid_cam.read()
    params['image_height'], params['image_width'], resized_channels = frame.shape

    # Compute the scaling factor to scale the image to a desired size
    if params['image_height'] != DESIRED_IMAGE_HEIGHT:
        params['scaling_factor'] = round((DESIRED_IMAGE_HEIGHT / params['image_height']), 2)  # Rounded scaling factor

    else:
        params['scaling_factor'] = 1

    print("params['scaling_factor']: ", params['scaling_factor'])

    # Compute resized width and height and resize the image
    params['resized_width'] = int(params['image_width'] * params['scaling_factor'])
    params['resized_height'] = int(params['image_height'] * params['scaling_factor'])
    dimension = (params['resized_width'], params['resized_height'])
    frame = cv2.resize(frame, dimension, interpolation=cv2.INTER_AREA)

    # Compute the center rectangle's half width and height
    params['cent_rect_half_width'] = round(params['resized_width'] * (0.5 * PERCENT_CENTER_RECT))  # Use half percent
    params['cent_rect_half_height'] = round(params['resized_height'] * (0.5 * PERCENT_CENTER_RECT))  # Use half percent

    # Compute the minimum target radius to follow. Smaller detected targets will be ignored
    params['min_tgt_radius'] = round(params['resized_width'] * PERCENT_TARGET_RADIUS)

    # Compute the position for the X and Y Cartesian coordinates in camera pixel units
    params['x_axe_pos'] = int(params['resized_height'] / 2 - 1)
    params['y_axe_pos'] = int(params['resized_width'] / 2 - 1)

    # Compute two points: p1 in the upper left and p2 in the lower right that will be used to
    # draw the center rectangle iin the image frame
    params['cent_rect_p1'] = (params['y_axe_pos'] - params['cent_rect_half_width'],
                              params['x_axe_pos'] - params['cent_rect_half_height'])
    params['cent_rect_p2'] = (params['y_axe_pos'] + params['cent_rect_half_width'],
                              params['x_axe_pos'] + params['cent_rect_half_height'])

    return


def get_target_coordinates(vid_cam):
    """ Detects a target by using color range segmentation and returns its 'camera pixel' coordinates."""

    # Lower and upper boundaries in HSV color space for the color we want to segment
    # Use the 'threshold_inRange.py' script included with the code to get
    # your own bounds with any color
    # To detect a blue target:

    HSV_LOWER_BOUND = (15, 5, 255)
    HSV_UPPER_BOUND = (44, 127, 255)
    
    # HSV_LOWER_BOUND = (66, 180, 18)
    # HSV_UPPER_BOUND = (120, 253, 221)

    # HSV_LOWER_BOUND = (22, 0, 0)
    # HSV_UPPER_BOUND = (180, 255, 255)

    # Grab a frame in BGR (Blue, Green, Red) space color
    is_grabbed, frame = vid_cam.read()

    # Resize the image frame for the detection process, if needed
    if params['scaling_factor'] != 1:
        dimension = (params['resized_width'], params['resized_height'])
        frame = cv2.resize(frame, dimension, interpolation=cv2.INTER_AREA)

    # Blur the image to remove high frequency content
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)

    # Change color space from BGR to HSV
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Get a mask with all the pixels inside our defined color boundaries
    mask = cv2.inRange(hsv, HSV_LOWER_BOUND, HSV_UPPER_BOUND)

    # Erode and dilate to remove small blobs
    mask = cv2.erode(mask, None, iterations=3)
    mask = cv2.dilate(mask, None, iterations=3)

    # Find all contours in the masked image
    contours, _ = cv2.findContours(mask,
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #_, contours, _= cv2.findContours(skin_ycrcb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Centroid coordinates to be returned:
    cX = None
    cY = None

    # To save the larges contour, presumably the detected object
    largest_contour = None

    # Check if at least one contour was found
    if len(contours) > 0:
        # Get the largest contour of all posibly detected
        largest_contour = max(contours, key=cv2.contourArea)

        # Compute the radius of an enclosing circle aorund the largest contour
        ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)

        # Compute centroid only if contour radius is larger than 0.5 half the center rectangle
        if radius > params['min_tgt_radius']:
            # Compute contour raw moments
            M = cv2.moments(largest_contour)
            # Get the contour's centroid
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

    # Return centroid coordinates (camera pixel units), the analized frame and the largest contour
    return {'width': cX, 'height': cY}, frame, largest_contour


def moving_average_filter(coord):
    """ Applies Low-Pass Moving Average Filter to a pair of (x, y) coordinates."""

    # Append new coordinates to filter buffers
    filt_buffer['width'].append(coord['width'])
    filt_buffer['height'].append(coord['height'])

    # If the filters were full already with a number of NUM_FILT_POINTS values, 
    # discard the oldest value (FIFO buffer)
    if len(filt_buffer['width']) > NUM_FILT_POINTS:
        filt_buffer['width'] = filt_buffer['width'][1:]
        filt_buffer['height'] = filt_buffer['height'][1:]

    # Compute filtered camera coordinates
    N = len(filt_buffer['width'])  # Get the number of values in buffers (will be < NUM_FILT_POINTS at the start)

    # Sum all values for each coordinate
    w_sum = sum(filt_buffer['width'])
    h_sum = sum(filt_buffer['height'])
    # Compute the average
    w_filt = int(round(w_sum / N))
    h_filt = int(round(h_sum / N))

    # Return filtered coordinates as a dictionary
    return {'width': w_filt, 'height': h_filt}


def draw_objects(cam_coord, filt_cam_coord, frame, contour): 
    """ Draws visualization objects from the detection process.
    Position coordinates of every object are always in 'camera pixel' units"""

    # Draw the Cartesian axes
    cv2.line(frame, (0, params['x_axe_pos']), (params['resized_width'], params['x_axe_pos']), (0, 128, 255), 1)
    cv2.line(frame, (params['y_axe_pos'], 0), (params['y_axe_pos'], params['resized_height']), (0, 128, 255), 1)
    cv2.circle(frame, (params['y_axe_pos'], params['x_axe_pos']), 1, (255, 255, 255), -1)

    # Draw the center (tolerance) rectangle
    cv2.rectangle(frame, params['cent_rect_p1'], params['cent_rect_p2'], (0, 178, 255), 1)

    # Draw the detected object's contour, if any
    if contour is not None:
        cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)

    # Compute Cartesian coordinates of unfiltered detected object's centroid
    x_cart_coord = cam_coord['width'] - params['y_axe_pos']
    y_cart_coord = params['x_axe_pos'] - cam_coord['height']

    # Compute Cartesian coordinates of filtered detected object's centroid
    x_filt_cart_coord = filt_cam_coord['width'] - params['y_axe_pos']
    y_filt_cart_coord = params['x_axe_pos'] - filt_cam_coord['height']

    # Draw unfiltered centroid as a red dot, including coordinate values
    cv2.circle(frame, (cam_coord['width'], cam_coord['height']), 5, (0, 0, 255), -1)
    cv2.putText(frame, str(x_cart_coord) + ", " + str(y_cart_coord),
                (cam_coord['width'] + 25, cam_coord['height'] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Draw filtered centroid as a blue dot, including coordinate values
    cv2.circle(frame, (filt_cam_coord['width'], filt_cam_coord['height']), 5, (255, 30, 30), -1)
    cv2.putText(frame, str(x_filt_cart_coord) + ", " + str(y_filt_cart_coord),
                (filt_cam_coord['width'] + 25, filt_cam_coord['height'] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 30, 30), 1)

    # Compute bounding box
    boundRect = cv2.boundingRect(contour)
    color = (0,0,255)
    cv2.rectangle(frame, (int(boundRect[0]), int(boundRect[1])), \
      (int(boundRect[0]+boundRect[2]), int(boundRect[1]+boundRect[3])), color, 2)
    cv2.putText(frame, "Fuego detectado!", (int(boundRect[0]), int(boundRect[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return frame  # Return the image frame with all drawn objects


# if __name__ == "__main__":
#     run()
