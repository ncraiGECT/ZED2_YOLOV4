from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet
import pyzed.sl as sl


def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax
def get_object_depth(depth, bounds):
    '''
    Calculates the median x, y, z position of top slice(area_div) of point cloud
    in camera frame.
    Arguments:
        depth: Point cloud data of whole frame.
        bounds: Bounding box for object in pixels.
            bounds[0]: x-center
            bounds[1]: y-center
            bounds[2]: width of bounding box.
            bounds[3]: height of bounding box.

    Return:
        x, y, z: Location of object in meters.
    '''
    area_div = 2

    x_vect = []
    y_vect = []
    z_vect = []

    for j in range(int(bounds[0] - area_div), int(bounds[0] + area_div)):
        for i in range(int(bounds[1] - area_div), int(bounds[1] + area_div)):
            z = depth[i, j, 2]
            if not np.isnan(z) and not np.isinf(z):
                x_vect.append(depth[i, j, 0])
                y_vect.append(depth[i, j, 1])
                z_vect.append(z)
    try:
        x_median = statistics.median(x_vect)
        y_median = statistics.median(y_vect)
        z_median = statistics.median(z_vect)
    except Exception:
        x_median = -1
        y_median = -1
        z_median = -1
        pass

    return x_median, y_median, z_median


def cvDrawBoxes(detections, img,distance):
    for detection in detections:
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        thickness = 1
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
        '''
        cv2.putText(img,  (str(distance) + " m"),
                            (pt1[0] + (thickness * 4), pt1[1] + (10 + thickness * 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)  '''
       
    return img


netMain = None
metaMain = None
altNames = None


def YOLO():

    global metaMain, netMain, altNames
    zed_id = 0
    configPath = "./cfg/yolov4.cfg"
    weightPath = "./yolov4.weights"
    metaPath = "./cfg/coco.data"
    input_type = sl.InputType()
    # Launch camera by id
    input_type.set_from_camera_id(zed_id)
    init = sl.InitParameters(input_t=input_type)
    init.coordinate_units = sl.UNIT.METER
    cam = sl.Camera()    
    status = cam.open(init)

    runtime = sl.RuntimeParameters()
    # Use STANDARD sensing mode
    runtime.sensing_mode = sl.SENSING_MODE.STANDARD
    mat = sl.Mat()
    point_cloud_mat = sl.Mat()
    
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
 
    print("Starting the YOLO loop...")


    darknet_image = darknet.make_image(darknet.network_width(netMain),
                                    darknet.network_height(netMain),3)
    while True:
        prev_time = time.time()
        
        err = cam.grab(runtime) 
        cam.retrieve_image(mat, sl.VIEW.LEFT)
        frame_read = mat.get_data()
        cam.retrieve_measure(point_cloud_mat, sl.MEASURE.XYZRGBA)
        depth = point_cloud_mat.get_data()
    
        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
     
        frame_resized = cv2.resize(frame_rgb,
                                   (darknet.network_width(netMain),
                                    darknet.network_height(netMain)),
                                   interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())

        detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)
        
        bounds = detections[1]
        print(bounds)
        x, y, z = get_object_depth(depth, bounds)
        x_coord = int(bounds[0] - bounds[2]/2)
        y_coord = int(bounds[1] - bounds[3]/2)
        distance = math.sqrt(x * x + y * y + z * z)
        #distance = "{:.2f}".format(distance)
        image = cvDrawBoxes(detections, frame_resized,distance)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        
        cv2.imshow('Demo', image)
        cv2.waitKey(3)
    
    

if __name__ == "__main__":
    YOLO()
