import streamlit as st
#import mediapipe as mp
import cv2
#import tempfile
#import time
import detect_py35 as dt

# picture = st.camera_input("Take a picture")

# if picture:
#     st.image(picture)
# dt.run()

@st.cache
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = width / float(w)
        dim = (int(w * r), height)

    else:
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    return resized

stframe = st.empty()


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
    dt.get_cam_params(vid_cam)
    # print(f"-- Original image width, height: {dt.params['image_width']}, {dt.params['image_height']}")

    # Infinite detect-follow loop
    while True:
        # Get the target coordinates (if any target was detected)
        tgt_cam_coord, frame, contour = dt.get_target_coordinates(vid_cam)

        # print('-- break point 1')
        # If a target was found, filter their coordinatesq
        if tgt_cam_coord['width'] is not None and tgt_cam_coord['height'] is not None:
            # Apply Moving Average filter to target camera coordinates
            tgt_filt_cam_coord = dt.moving_average_filter(tgt_cam_coord)

        # No target was found, set target camera coordinates to the Cartesian origin,
        # so the drone doesn't move
        else:
            # The Cartesian origin is where the x and y Cartesian axes are located
            # in the image, in pixel units
            tgt_cam_coord = {'width': dt.params['y_axe_pos'],
                             'height': dt.params['x_axe_pos']}  # Needed just for drawing objects
            tgt_filt_cam_coord = {'width': dt.params['y_axe_pos'], 'height': dt.params['x_axe_pos']}

        # Convert from camera coordinates to Cartesian coordinates (in pixel units)
        tgt_cart_coord = {'x': (tgt_filt_cam_coord['width'] - dt.params['y_axe_pos']),
                          'y': (dt.params['x_axe_pos'] - tgt_filt_cam_coord['height'])}

        # Compute scaling conversion factor from camera coordinates in pixel units
        # to Cartesian coordinates in meters
        COORD_SYS_CONV_FACTOR = 0.1

        # If the target is outside the center rectangle, compute North and East coordinates 
        if abs(tgt_cart_coord['x']) > dt.params['cent_rect_half_width'] or \
                abs(tgt_cart_coord['y']) > dt.params['cent_rect_half_height']:
            # Compute North, East coordinates applying "camera pixel" to Cartesian conversion factor
            E_coord = tgt_cart_coord['x'] * COORD_SYS_CONV_FACTOR
            N_coord = tgt_cart_coord['y'] * COORD_SYS_CONV_FACTOR
            # D_coord, yaw_angle don't change

        # Draw objects over the detection image frame just for visualization
        frame = dt.draw_objects(tgt_cam_coord, tgt_filt_cam_coord, frame, contour)

        # Show the detection image frame on screen
        cv2.imshow("Detection of fire", frame)

        # my_placeholder = st.empty()
        # my_placeholder.image(frame, use_column_width=True)

        # st.write(frame.shape)
        # st.image(my_placeholder)

        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # st.image(frame)
        # st.write(frame)
        
        # frame = cv2.resize(frame, (0, 0), fx=0.8, fy=0.8)
        # frame = image_resize(image=frame, width=640)
        stframe.image(frame, channels = 'BGR', use_column_width = True)

        # Catch aborting key from computer keyboard
        key = cv2.waitKey(1) & 0xFF
        # If the 'q' key is pressed, break the 'while' infinite loop
        if key == ord("q"):
            break

    print("The script has ended.")

run()