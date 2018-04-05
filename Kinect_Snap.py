"""
18.04.05 version
"""
import PyKinectV2
from PyKinectV2 import *
import PyKinectRuntime
import numpy as np
import cv2
import ctypes
import time

calib_mat = np.array([[-0.983879,  0.022774,  0.005712, -0.080417],
                      [ 0.018603,  0.993666, -0.085626, -0.032837],
                      [ 0.009604, -0.075230, -0.957587,  0.935117],
                      [ 0.0,       0.0,       0.0,       1.000000]])


class Kinect(object):
    def __init__(self):
        self._kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Depth | PyKinectV2.FrameSourceTypes_Infrared)
        self.depth_frame = np.empty([512, 424])
        self.focalLength = 1485.73
        self.centerX = 960.5
        self.centerY = 540.5
        self.lower = np.array([180, 135, 64], dtype="uint8")
        self.upper = np.array([244, 188, 99], dtype="uint8")

    def snap(self):
        result_img = self.scene()
        cut_img = result_img[164:,]

        return cut_img

    def scene(self):
        while True:
            if self._kinect.has_new_color_frame():
                raw_array = self._kinect.get_last_color_frame()
                raw_img = raw_array.reshape((1080, 1920, 4))                           # to Mat

                cropped_img = cv2.flip(raw_img[33:912, 540:1329], 1)                                     # Flipped Image

                resized_img = cv2.resize(cropped_img, (256, 256))                       # Resized image (256,256) RGBA
                result_img = cv2.cvtColor(resized_img, cv2.COLOR_RGBA2RGB)              # Format : RGB
                break

        return result_img

    def color2xyz(self, data):
        while True:
            time.sleep(0.01)
            if self._kinect.has_new_depth_frame():
                depth_frame = self._kinect.get_last_depth_frame()
                break

        mean_pxl = np.mean(data, axis=0).astype(np.uint32)

        # Make patch, detected object center pixel.
        pxl_patch = []
        start_pxl = mean_pxl - np.array([2, 2])

        for i in range(5):
            for j in range(5):
                pxl_patch.append(start_pxl + np.array([i, j]))

        # Pixel Revision
        for idx, [y, x] in enumerate(pxl_patch):
            pxl_patch[idx][0] = 3.08203125 * (255 - x) + 540  # Width revision, rate : 3, offset : 521
            pxl_patch[idx][1] = 3.43359375 * (y + 128) + 33  # Height revision, rate : 3.246, offset : 108

        pxl_patch = np.array(pxl_patch).astype(np.uint32)  # round? int?

        p_b = ctypes.pointer((_CameraSpacePoint * 1920 * 1080)())
        p_a = ctypes.cast(p_b, ctypes.POINTER(_CameraSpacePoint))

        depth_size = 512 * 424
        color_size = 1920 * 1080
        c_depthframe = np.ctypeslib.as_ctypes(depth_frame)
        p_depthframe1 = ctypes.pointer(c_depthframe)
        p_depthframe = ctypes.cast(p_depthframe1, ctypes.POINTER(ctypes.c_ushort))

        self._kinect._mapper.MapColorFrameToCameraSpace(depth_size, p_depthframe, color_size, p_a)

        camera_space_point = np.ctypeslib.as_array((ctypes.c_float * 1920 * 1080 * 3).from_address(ctypes.addressof(p_a.contents)))
        d = np.reshape(camera_space_point, (1, np.product(camera_space_point.shape)))
        xyz_frame = np.reshape(d, (1080, 1920, 3))

        xyz_list = []

        for idx, [x, y] in enumerate(pxl_patch):
            kinect_pos = np.array([[xyz_frame[y, x, 0], xyz_frame[y, x, 1], xyz_frame[y, x, 2], 1]])
            xyz = np.dot(calib_mat, kinect_pos.T)
            xyz = xyz.flatten()[:-1]
            xyz_list.append(xyz)

        xyz_list = np.array(xyz_list)

        # Delete nan value
        if np.any(np.isnan(xyz_list.flatten())):
            nan_idx = np.sort(np.transpose(np.argwhere(np.isnan(xyz_list))[0::3])[0])
            for x in reversed(nan_idx):
                xyz_list = np.delete(xyz_list, x, 0)

        mean_xyz = np.mean(xyz_list, axis=0)

        if np.any(np.isinf(mean_xyz)) or np.any(np.isnan(mean_xyz)):
            return None
        else:
            return mean_xyz
