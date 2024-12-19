#!/usr/bin/env python2.7

import rospy 
import os
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from message_filters import Subscriber, ApproximateTimeSynchronizer
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, ColorRGBA
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, Point32, PolygonStamped

import cv2
import sys
import math
import numpy as np
import signal

from openpose import pyopenpose as op
from time import sleep
from multiprocessing import Process, Pipe, Value

bridge = CvBridge()

# Params from launch file
use_stereo = False
depth_neighborhood = 5
pipe_camera_recv, pipe_camera_send = Pipe(duplex=False)
pipe_openpose_recv, pipe_openpose_send = Pipe(duplex=False)

params = dict()
#params["logging_level"] = 3
#params["camera"]=-1
#params["output_resolution"] = "-1x-1"

params["model_pose"] = "BODY_25"
params["disable_multi_thread"] = True
params["render_threshold"] = 0.0
params["process_real_time"] = True
params["model_folder"] = ""

is_running = Value('b', True)

HEAD = 0
SPINE = 1
SPINE_BASE = 8

SHOULDER_R = 2
ELBOW_R = 3
WRIST_R = 4
HIP_R = 9
KNEE_R = 10
FOOT_R = 11

SHOULDER_L = 5
ELBOW_L = 6
WRIST_L = 7
HIP_L = 12
KNEE_L = 13
FOOT_L = 14


class CameraData:
    def __init__(self):
        self.frame_id = None
        self.stamp = None
        self.image_rgb_left = None
        self.image_rgb_right = None
        self.image_depth = None
        self.fx_depth = None
        self.fy_depth = None
        self.cx_depth = None
        self.cy_depth = None
        self.baseline_m = None
        self.ratio_depth_to_rgb_x = None
        self.ratio_depth_to_rgb_y = None


def get_nanmedian_z(u, v, neighborhood, camera_data):

    height, width = camera_data.image_depth.shape

    v0 = np.maximum(0, v - neighborhood)
    v1 = np.minimum(height - 1, v + neighborhood)
    u0 = np.maximum(0, u - neighborhood)
    u1 = np.minimum(width - 1, u + neighborhood)

    z_median = np.nanmedian(camera_data.image_depth[int(v0):int(v1), int(u0):int(u1)])

    if np.isnan(z_median):
        z_median = get_nanmedian_z(u, v, neighborhood * 1.5, camera_data)

    return z_median


def get_xyz(keypoint, camera_data):
    u = keypoint[0]
    v = keypoint[1]
    u_depth = u * camera_data.ratio_depth_to_rgb_x
    v_depth = v * camera_data.ratio_depth_to_rgb_y
    if (u == 0) and (v == 0):
        z = float('nan')
        x = float('nan')
        y = float('nan')
    else:
        z = get_nanmedian_z(u_depth, v_depth, depth_neighborhood, camera_data)
        x = (u_depth - camera_data.cx_depth) * z / camera_data.fx_depth
        y = (v_depth - camera_data.cy_depth) * z / camera_data.fy_depth
    return x, y, z


def get_xyz_stereo(keypoint_left, keypoint_right, camera_data):
    u_left = keypoint_left[0]
    v_left = keypoint_left[1]
    u_right = keypoint_right[0]
    if (u_left == 0) and (v_left == 0):
        z = float('nan')
        x = float('nan')
        y = float('nan')
    else:
        z = camera_data.fx_depth / camera_data.ratio_depth_to_rgb_x * camera_data.baseline_m / (u_left - u_right)
        x = (u_left - camera_data.cx_depth / camera_data.ratio_depth_to_rgb_x) * camera_data.baseline_m / (u_left - u_right)
        y = (v_left - camera_data.cy_depth / camera_data.ratio_depth_to_rgb_y) * camera_data.baseline_m / (u_left - u_right)
    return x, y, z


def get_keypoint_as_Point32(keypoint, camera_data):
    pt = Point32()
    pt.x, pt.y, pt.z = get_xyz(keypoint, camera_data)
    return pt


def get_keypoint_as_Point32_stereo(keypoint_left, keypoint_right, camera_data):
    pt = Point32()
    pt.x, pt.y, pt.z = get_xyz_stereo(keypoint_left, keypoint_right, camera_data)
    return pt


def draw_info(output_image, u, v, angle_deg, distance):

    offset = 5

    font_face = cv2.FONT_HERSHEY_DUPLEX
    scale = 0.8
    thickness = 2
    text = "%.2f" % angle_deg
    text_size, baseline = cv2.getTextSize(text, font_face, scale, thickness)

    cv2.putText(output_image, text, (int(u) + offset, int(v) + offset + text_size[1]),
                font_face, scale, (0, 0, 255), thickness, lineType=cv2.LINE_AA)

    scale = 0.6
    thickness = 1
    text = "Elbow angle = %.2f" % angle_deg
    text_size, baseline = cv2.getTextSize(text, font_face, scale, thickness)
    cv2.putText(output_image, text, (offset, offset + text_size[1]),
                font_face, scale, (0, 0, 255), thickness, lineType=cv2.LINE_AA)

    text = "Distance [m] = %.2f" % distance
    cv2.putText(output_image, text, (offset, 2 * (offset + text_size[1])),
                font_face, scale, (0, 0, 255), thickness, lineType=cv2.LINE_AA)


def function_openpose(pipe_camera, pipe_openpose, is_running):
    openpose = op.WrapperPython()
    openpose.configure(params)
    openpose.start()

    while is_running.value:
        camera_data = pipe_camera.recv()[0]

        # Process image
        datum = op.Datum()
        datum.cvInputData = camera_data.image_rgb_left
        openpose.emplaceAndPop([datum])

        # If nothing was recognized
        if datum.poseKeypoints.size == 1:
            continue

        if use_stereo:
            datum_right = op.Datum()
            datum_right.cvInputData = camera_data.image_rgb_right
            openpose.emplaceAndPop([datum_right])
            # If nothing was recognized
            if (datum_right.poseKeypoints.size == 1) or \
                (datum.poseKeypoints.size != datum_right.poseKeypoints.size):
                continue

        keypoints_left = datum.poseKeypoints
        output_image = datum.cvOutputData.copy()

        if use_stereo:
            keypoints_right = datum_right.poseKeypoints
            output_image_right = datum_right.cvOutputData.copy()

        nearest_body = -1
        distance_min = sys.float_info.max

        for body_index in xrange(0, len(keypoints_left)):
            if use_stereo:
                x_shoulder_l, y_shoulder_l, z_shoulder_l = get_xyz_stereo(keypoints_left[body_index, SHOULDER_L],
                                                                          keypoints_right[body_index, SHOULDER_L],
                                                                          camera_data)
                x_shoulder_r, y_shoulder_r, z_shoulder_r = get_xyz_stereo(keypoints_left[body_index, SHOULDER_R],
                                                                          keypoints_right[body_index, SHOULDER_R],
                                                                          camera_data)
            else:
                x_shoulder_l, y_shoulder_l, z_shoulder_l = get_xyz(keypoints_left[body_index, SHOULDER_L], camera_data)
                x_shoulder_r, y_shoulder_r, z_shoulder_r = get_xyz(keypoints_left[body_index, SHOULDER_R], camera_data)

            x_center = (x_shoulder_l + x_shoulder_r) / 2.0
            y_center = (y_shoulder_l + y_shoulder_r) / 2.0
            z_center = (z_shoulder_l + z_shoulder_r) / 2.0
            distance = x_center * x_center + y_center * y_center + z_center * z_center

            if (distance > 0.0) and (distance < distance_min):
                distance_min = distance
                nearest_body = body_index

        distance_min = math.sqrt(distance_min)

        # Calculate the right hand elbow angle formed between R-shoulder and R-wrist
        # Keypoints: https://www.researchgate.net/figure/An-example-of-the-skeleton-representation-obtained-using-the-OpenPose-library-a-shows_fig1_326111194
        if nearest_body >= 0:
            if use_stereo:
                x_elbow_r, y_elbow_r, z_elbow_r = get_xyz_stereo(keypoints_left[nearest_body, ELBOW_R],
                                                                 keypoints_right[nearest_body, ELBOW_R], camera_data)
                x_shoulder_r, y_shoulder_r, z_shoulder_r = get_xyz_stereo(keypoints_left[nearest_body, SHOULDER_R],
                                                                          keypoints_right[nearest_body, SHOULDER_R],
                                                                          camera_data)
                x_wrist_r, y_wrist_r, z_wrist_r = get_xyz_stereo(keypoints_left[nearest_body, WRIST_R],
                                                                 keypoints_right[nearest_body, WRIST_R], camera_data)
            else:
                x_elbow_r, y_elbow_r, z_elbow_r = get_xyz(keypoints_left[nearest_body, ELBOW_R], camera_data)
                x_shoulder_r, y_shoulder_r, z_shoulder_r = get_xyz(keypoints_left[nearest_body, SHOULDER_R],
                                                                   camera_data)
                x_wrist_r, y_wrist_r, z_wrist_r = get_xyz(keypoints_left[nearest_body, WRIST_R], camera_data)

            vec_elbow_r = np.array([x_elbow_r, y_elbow_r, z_elbow_r])
            vec_shoulder_r = np.array([x_shoulder_r, y_shoulder_r, z_shoulder_r])
            vec_wrist_r = np.array([x_wrist_r, y_wrist_r, z_wrist_r])

            vec_forearm = vec_wrist_r - vec_elbow_r
            vec_arm = vec_shoulder_r - vec_elbow_r

            norm = np.linalg.norm(vec_forearm) * np.linalg.norm(vec_arm)

            if norm < 0.0000001:
                angle_rad = 0.0
            else:
                angle_rad = np.arccos(np.dot(vec_forearm, vec_arm) / norm)
            angle_deg = 180.0 * angle_rad / np.pi

            u_elbow_r = keypoints_left[nearest_body, ELBOW_R, 0]
            v_elbow_r = keypoints_left[nearest_body, ELBOW_R, 1]

            #rospy.loginfo("Elbow angle [deg] = %.2f; distance = %.2f", angle_deg, distance_min)
            draw_info(output_image, u_elbow_r, v_elbow_r, angle_deg, distance_min)

            angle_deg = 0.1
            if angle_deg > 0.0000001:
                values = Float32MultiArray()
                values.layout.dim.append(MultiArrayDimension())
                values.layout.dim[0].label = "values"
                values.layout.dim[0].size = 2
                values.layout.dim[0].stride = 2
                values.layout.data_offset = 0
                values.data = [angle_deg, distance_min]

                points3d = PolygonStamped()
                points3d.polygon.points = []
                for i in xrange(0, len(keypoints_left[nearest_body])):
                    if use_stereo:
                        keypoint32 = get_keypoint_as_Point32_stereo(keypoints_left[nearest_body, i],
                                                                    keypoints_right[nearest_body, i], camera_data)
                    else:
                        keypoint32 = get_keypoint_as_Point32(keypoints_left[nearest_body, i], camera_data)

                    points3d.polygon.points.append(keypoint32)

                # u = int(keypoints_left[nearest_body, FOOT_R, 0])
                # v = int(keypoints_left[nearest_body, FOOT_R, 1])
                # cv2.rectangle(output_image, (u - 20, v - 20), (u + 20, v + 20), (255, 0, 0), 3)

                image_msg = bridge.cv2_to_imgmsg(output_image, encoding="passthrough")

                if use_stereo:
                    image_right_msg = bridge.cv2_to_imgmsg(output_image_right, encoding="passthrough")
                else:
                    image_right_msg = image_msg

                image_msg.header.frame_id = camera_data.frame_id
                image_msg.header.stamp = camera_data.stamp

                points3d.header.frame_id = camera_data.frame_id
                points3d.header.stamp = camera_data.stamp

                image_right_msg.header.frame_id = camera_data.frame_id
                image_right_msg.header.stamp = camera_data.stamp

                pipe_openpose.send([image_msg, values, points3d, image_right_msg])


def create_marker_msgs(keypoints_msg):
    links_msg = Marker()
    links_msg.header.frame_id = keypoints_msg.header.frame_id
    links_msg.header.stamp = keypoints_msg.header.stamp
    links_msg.type = links_msg.LINE_LIST
    links_msg.action = links_msg.ADD
    links_msg.pose.orientation.w = 1
    links_msg.scale.x = 0.01
    links_msg.scale.y = 0.01
    links_msg.scale.z = 0.01

    links_msg.points = []
    links_msg.colors = []

    color = ColorRGBA()
    color.r = 0.0
    color.g = 1.0
    color.b = 0.0
    color.a = 1.0

    pt1 = keypoints_msg.polygon.points[SHOULDER_R]
    pt2 = keypoints_msg.polygon.points[SHOULDER_L]
    if not math.isnan(pt2.x) and not math.isnan(pt2.y) and not math.isnan(pt2.z):
        links_msg.points.append(pt1)
        links_msg.colors.append(color)
        links_msg.points.append(pt2)
        links_msg.colors.append(color)
    
    pt1 = keypoints_msg.polygon.points[SHOULDER_R]
    pt2 = keypoints_msg.polygon.points[ELBOW_R]
    if not math.isnan(pt2.x) and not math.isnan(pt2.y) and not math.isnan(pt2.z):
        links_msg.points.append(pt1)
        links_msg.colors.append(color)
        links_msg.points.append(pt2)
        links_msg.colors.append(color)

    pt1 = keypoints_msg.polygon.points[ELBOW_R]
    pt2 = keypoints_msg.polygon.points[WRIST_R]
    if not math.isnan(pt2.x) and not math.isnan(pt2.y) and not math.isnan(pt2.z):
        links_msg.points.append(pt1)
        links_msg.colors.append(color)
        links_msg.points.append(pt2)
        links_msg.colors.append(color)

    pt1 = keypoints_msg.polygon.points[SHOULDER_L]
    pt2 = keypoints_msg.polygon.points[ELBOW_L]
    if not math.isnan(pt2.x) and not math.isnan(pt2.y) and not math.isnan(pt2.z):
        links_msg.points.append(pt1)
        links_msg.colors.append(color)
        links_msg.points.append(pt2)
        links_msg.colors.append(color)

    pt1 = keypoints_msg.polygon.points[ELBOW_L]
    pt2 = keypoints_msg.polygon.points[WRIST_L]
    if not math.isnan(pt2.x) and not math.isnan(pt2.y) and not math.isnan(pt2.z):
        links_msg.points.append(pt1)
        links_msg.colors.append(color)
        links_msg.points.append(pt2)
        links_msg.colors.append(color)

    pt1 = keypoints_msg.polygon.points[SPINE]
    pt2 = keypoints_msg.polygon.points[SPINE_BASE]
    if not math.isnan(pt2.x) and not math.isnan(pt2.y) and not math.isnan(pt2.z):
        links_msg.points.append(pt1)
        links_msg.colors.append(color)
        links_msg.points.append(pt2)
        links_msg.colors.append(color)

    pt1 = keypoints_msg.polygon.points[SPINE_BASE]
    pt2 = keypoints_msg.polygon.points[HIP_L]
    if not math.isnan(pt2.x) and not math.isnan(pt2.y) and not math.isnan(pt2.z):
        links_msg.points.append(pt1)
        links_msg.colors.append(color)
        links_msg.points.append(pt2)
        links_msg.colors.append(color)

    pt1 = keypoints_msg.polygon.points[HIP_L]
    pt2 = keypoints_msg.polygon.points[KNEE_L]
    if not math.isnan(pt2.x) and not math.isnan(pt2.y) and not math.isnan(pt2.z):
        links_msg.points.append(pt1)
        links_msg.colors.append(color)
        links_msg.points.append(pt2)
        links_msg.colors.append(color)

    pt1 = keypoints_msg.polygon.points[KNEE_L]
    pt2 = keypoints_msg.polygon.points[FOOT_L]
    if not math.isnan(pt2.x) and not math.isnan(pt2.y) and not math.isnan(pt2.z):
        links_msg.points.append(pt1)
        links_msg.colors.append(color)
        links_msg.points.append(pt2)
        links_msg.colors.append(color)

    pt1 = keypoints_msg.polygon.points[SPINE_BASE]
    pt2 = keypoints_msg.polygon.points[HIP_R]
    if not math.isnan(pt2.x) and not math.isnan(pt2.y) and not math.isnan(pt2.z):
        links_msg.points.append(pt1)
        links_msg.colors.append(color)
        links_msg.points.append(pt2)
        links_msg.colors.append(color)

    pt1 = keypoints_msg.polygon.points[HIP_R]
    pt2 = keypoints_msg.polygon.points[KNEE_R]
    if not math.isnan(pt2.x) and not math.isnan(pt2.y) and not math.isnan(pt2.z):
        links_msg.points.append(pt1)
        links_msg.colors.append(color)
        links_msg.points.append(pt2)
        links_msg.colors.append(color)

    pt1 = keypoints_msg.polygon.points[KNEE_R]
    pt2 = keypoints_msg.polygon.points[FOOT_R]
    if not math.isnan(pt2.x) and not math.isnan(pt2.y) and not math.isnan(pt2.z):
        links_msg.points.append(pt1)
        links_msg.colors.append(color)
        links_msg.points.append(pt2)
        links_msg.colors.append(color)

    t = rospy.Duration()
    links_msg.lifetime = t

    joints_msg = Marker()
    joints_msg.header.frame_id = keypoints_msg.header.frame_id
    joints_msg.header.stamp = keypoints_msg.header.stamp
    joints_msg.type = joints_msg.SPHERE_LIST
    joints_msg.action = joints_msg.ADD
    joints_msg.pose.orientation.w = 1
    joints_msg.scale.x = 0.05
    joints_msg.scale.y = 0.05
    joints_msg.scale.z = 0.05
    #marker.color.a = 1.0
    #marker.color.r = 1.0
    
    joints_msg.points = []
    joints_msg.colors = []

    color = ColorRGBA()
    color.r = 0.0
    color.g = 1.0
    color.b = 1.0
    color.a = 1.0

    pt = keypoints_msg.polygon.points[SPINE]
    joints_msg.points.append(pt)
    joints_msg.colors.append(color)

    color = ColorRGBA()
    color.r = 1.0
    color.g = 1.0
    color.b = 1.0
    color.a = 1.0

    pt = keypoints_msg.polygon.points[HEAD]
    joints_msg.points.append(pt)
    joints_msg.colors.append(color)

    color = ColorRGBA()
    color.r = 1.0
    color.g = 0.0
    color.b = 0.0
    color.a = 1.0

    pt = keypoints_msg.polygon.points[SHOULDER_R]
    if not math.isnan(pt.x) and not math.isnan(pt.y) and not math.isnan(pt.z):
        joints_msg.points.append(pt)
        joints_msg.colors.append(color)

    pt = keypoints_msg.polygon.points[ELBOW_R]
    if not math.isnan(pt.x) and not math.isnan(pt.y) and not math.isnan(pt.z):
        joints_msg.points.append(pt)
        joints_msg.colors.append(color)

    pt = keypoints_msg.polygon.points[WRIST_R]
    if not math.isnan(pt.x) and not math.isnan(pt.y) and not math.isnan(pt.z):
        joints_msg.points.append(pt)
        joints_msg.colors.append(color)

    pt = keypoints_msg.polygon.points[HIP_R]
    if not math.isnan(pt.x) and not math.isnan(pt.y) and not math.isnan(pt.z):
        joints_msg.points.append(pt)
        joints_msg.colors.append(color)

    pt = keypoints_msg.polygon.points[KNEE_R]
    if not math.isnan(pt.x) and not math.isnan(pt.y) and not math.isnan(pt.z):
        joints_msg.points.append(pt)
        joints_msg.colors.append(color)

    pt = keypoints_msg.polygon.points[FOOT_R]
    if not math.isnan(pt.x) and not math.isnan(pt.y) and not math.isnan(pt.z):
        joints_msg.points.append(pt)
        joints_msg.colors.append(color)

    color = ColorRGBA()
    color.r = 0.0
    color.g = 1.0
    color.b = 1.0
    color.a = 1.0

    pt = keypoints_msg.polygon.points[SPINE]
    joints_msg.points.append(pt)
    joints_msg.colors.append(color)

    pt = keypoints_msg.polygon.points[SPINE_BASE]
    joints_msg.points.append(pt)
    joints_msg.colors.append(color)

    color = ColorRGBA()
    color.r = 0.0
    color.g = 1.0
    color.b = 0.0
    color.a = 1.0

    pt = keypoints_msg.polygon.points[SHOULDER_L]
    if not math.isnan(pt.x) and not math.isnan(pt.y) and not math.isnan(pt.z):
        joints_msg.points.append(pt)
        joints_msg.colors.append(color)

    pt = keypoints_msg.polygon.points[ELBOW_L]
    if not math.isnan(pt.x) and not math.isnan(pt.y) and not math.isnan(pt.z):
        joints_msg.points.append(pt)
        joints_msg.colors.append(color)

    pt = keypoints_msg.polygon.points[WRIST_L]
    if not math.isnan(pt.x) and not math.isnan(pt.y) and not math.isnan(pt.z):
        joints_msg.points.append(pt)
        joints_msg.colors.append(color)

    pt = keypoints_msg.polygon.points[HIP_L]
    if not math.isnan(pt.x) and not math.isnan(pt.y) and not math.isnan(pt.z):
        joints_msg.points.append(pt)
        joints_msg.colors.append(color)

    pt = keypoints_msg.polygon.points[KNEE_L]
    if not math.isnan(pt.x) and not math.isnan(pt.y) and not math.isnan(pt.z):
        joints_msg.points.append(pt)
        joints_msg.colors.append(color)

    pt = keypoints_msg.polygon.points[FOOT_L]
    if not math.isnan(pt.x) and not math.isnan(pt.y) and not math.isnan(pt.z):
        joints_msg.points.append(pt)
        joints_msg.colors.append(color)

    t = rospy.Duration()
    joints_msg.lifetime = t
    return joints_msg, links_msg


def callback(image_rgb_left_msg, image_rgb_right_msg, image_depth_msg, camera_info_left_msg, camera_info_right_msg):
    global pipe_camera_send
    global use_stereo

    # sys.stdout.write('.')
    # sys.stdout.flush()

    try:
        rospy.loginfo("Camera callback start")
        camera_data = CameraData()

        camera_data.frame_id = image_rgb_left_msg.header.frame_id
        camera_data.stamp = image_rgb_left_msg.header.stamp
        camera_data.image_rgb_left = bridge.imgmsg_to_cv2(image_rgb_left_msg, "bgr8").copy()
        camera_data.image_depth = bridge.imgmsg_to_cv2(image_depth_msg, "32FC1").copy()

        if use_stereo:
            camera_data.image_rgb_right = bridge.imgmsg_to_cv2(image_rgb_right_msg, "bgr8").copy()
            camera_data.baseline_m = -camera_info_right_msg.P[3] / camera_info_right_msg.P[0]

        height_depth, width_depth = camera_data.image_depth.shape
        height_rgb, width_rgb, channels = camera_data.image_rgb_left.shape

        camera_data.ratio_depth_to_rgb_x = width_depth / float(width_rgb)
        camera_data.ratio_depth_to_rgb_y = height_depth / float(height_rgb)

        camera_data.fx_depth = camera_info_left_msg.P[0] * camera_data.ratio_depth_to_rgb_x
        camera_data.fy_depth = camera_info_left_msg.P[5] * camera_data.ratio_depth_to_rgb_y
        camera_data.cx_depth = camera_info_left_msg.P[2] * camera_data.ratio_depth_to_rgb_x
        camera_data.cy_depth = camera_info_left_msg.P[6] * camera_data.ratio_depth_to_rgb_y

        rospy.loginfo("Camera callback end")
        pipe_camera_send.send([camera_data])
    except CvBridgeError as e:
        print(e)


def main_node():
    global use_stereo
    global is_running
    global depth_neighborhood

    global pipe_camera_recv
    global pipe_openpose_recv
    global pipe_openpose_send
    
    rospy.init_node('human_body_detection_node')

    use_stereo = rospy.get_param("~use_stereo")
    depth_neighborhood = rospy.get_param("~depth_neighborhood")
    accuracy_speed_ratio = rospy.get_param("~accuracy_speed_ratio")
    default_model_folder = rospy.get_param("~models_pathname")
    
    topic_image_rgb = rospy.get_param("~topic_image_rgb_left")
    topic_image_rgb_right = rospy.get_param("~topic_image_rgb_right")
    topic_image_depth = rospy.get_param("~topic_image_depth")
    topic_camera_info = rospy.get_param("~topic_camera_info_left")
    topic_camera_info_right = rospy.get_param("~topic_camera_info_right")

    params["model_folder"] = default_model_folder
    # Best accuracy
    if accuracy_speed_ratio == 0:
        params["net_resolution"] = "1312x736"
        params["scale_gap"] = 0.25
        params["scale_number"] = 4
    # Optimal speed and accuracy
    elif accuracy_speed_ratio == 1:
        params["net_resolution"] = "-1x368"
        params["scale_gap"] = 0.3
        params["scale_number"] = 1
    # Best speed
    elif accuracy_speed_ratio == 2:
        params["net_resolution"] = "-1x80"
        params["scale_gap"] = 0.3
        params["scale_number"] = 1

    pub_image = rospy.Publisher(rospy.get_name() + '/image_left', Image, queue_size=1)
    pub_values = rospy.Publisher(rospy.get_name() + '/values', Float32MultiArray, queue_size=1)
    pub_points3d = rospy.Publisher(rospy.get_name() + '/points', PolygonStamped, queue_size=1)
    pub_joints = rospy.Publisher(rospy.get_name() + '/joints', Marker, queue_size=100)
    pub_links = rospy.Publisher(rospy.get_name() + '/links', Marker, queue_size=100)
    pub_image_right = rospy.Publisher(rospy.get_name() + '/image_right', Image, queue_size=1)

    sub_rgb_left = Subscriber(topic_image_rgb, Image)
    sub_depth = Subscriber(topic_image_depth, Image)
    sub_camera_info_left = Subscriber(topic_camera_info, CameraInfo)

    if use_stereo:
        sub_rgb_right = Subscriber(topic_image_rgb_right, Image)
        sub_camera_info_right = Subscriber(topic_camera_info_right, CameraInfo)
    else:
        sub_rgb_right = sub_rgb_left
        sub_camera_info_right = sub_camera_info_left

    ats = ApproximateTimeSynchronizer(
        [sub_rgb_left, sub_rgb_right, sub_depth, sub_camera_info_left, sub_camera_info_right], queue_size=30,
        slop=0.2) #og = 0.03
    ats.registerCallback(callback)

    process_openpose = Process(target=function_openpose, args=(pipe_camera_recv, pipe_openpose_send, is_running,))
    process_openpose.start()

    #rospy.spin()
    while not rospy.is_shutdown():
        openpose_data = pipe_openpose_recv.recv()

        joints_msg, links_msg = create_marker_msgs(openpose_data[2])

        pub_image.publish(openpose_data[0])
        pub_values.publish(openpose_data[1])
        pub_points3d.publish(openpose_data[2])
        pub_links.publish(links_msg)
        pub_joints.publish(joints_msg)
        pub_image_right.publish(openpose_data[3])

    is_running.value = False
    print "lurb"
    process_openpose.join()


def receiveSignal(signalNumber, frame):
    global is_running
    is_running.value = False
    return

def shutdown_hook():
    print "Shutdown "
    os.kill(os.getpid(), 9)

if __name__ == '__main__':
    signal.signal(signal.SIGTERM, receiveSignal)
    signal.signal(signal.SIGINT, receiveSignal)
    #rospy.on_shutdown(shutdown_hook)
    try:
        main_node()
    except rospy.ROSInterruptException:
        pass
