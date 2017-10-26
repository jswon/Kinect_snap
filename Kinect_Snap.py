import PyKinectV2
from PyKinectV2 import *
import PyKinectRuntime
import numpy as np
import cv2
import ctypes
import time


class Kinect(object):
    def __init__(self):
        self._kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Depth | PyKinectV2.FrameSourceTypes_Infrared)
        self.depth_frame = np.empty([512, 424])
        self.focalLength = 1485.73
        self.centerX = 960.5
        self.centerY = 540.5

    def snap(self):
        while True:
            if self._kinect.has_new_color_frame():
                raw_array = self._kinect.get_last_color_frame()

                # TODO Camera Position confirm.
                raw_img = raw_array.reshape((1080, 1920, 4))                           # to Mat
                flipped_img = cv2.flip(raw_img, 1)                                     # Flipped Image
                cropped_img = flipped_img[143:961, 573:1350]                           # cropped ROI, Global View
                result_img = cv2.resize(cropped_img, (256, 256))                       # Resized image (256,256) RGBA
                result_img = cv2.cvtColor(result_img, cv2.COLOR_RGBA2RGB)              # Format : RGB
                break

        while True:
            if self._kinect.has_new_depth_frame():
                depth_frame = self._kinect.get_last_depth_frame()
                break

        return result_img[156:, ], depth_frame

    def color2xyz(self, depth_frame, data):
        """
        :param depth_frame:
        :param data: y,x pixel of index
        :return:
        """
        # while True:
        #     if self._kinect.has_new_depth_frame():
        #         depth_frame = self._kinect.get_last_depth_frame()
        #         break
        #     else:
        #         time.sleep(2)
        # depth_frame = self.depth_frame
        # calib_mat = np.array([[-0.929959, -0.001979, 0.077564, -0.130750], [0.003083, 0.937452, -0.094629, 0.058237], [-0.011651, 0.020211, -1.148202, 1.268683],
        #                       [0.0, 0.0, 0.0, 1.0000]])  # until 201 data
        calib_Mat = np.array([[-0.934119, 0.004855, -0.019816, -0.026845],
                              [0.011966, 0.942211, -0.158447, 0.128980],
                              [-0.010695, 0.014957, -1.304078, 1.441495],
                              [0.0, 0.0, 0.0, 1.0]])  # until 201 data

        target = np.array([data[1], data[0]])

        # Revision for resize(256
        target[0] = (256 - target[0]) * 3.035 + 573   # Width revision, rate : 3.03515625, offset : 573
        target[1] = (target[1] + 128) * 3.1953125 + 143       # Height revision, rate : 3.1953125, offset : 143

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

        target = target.astype(np.uint32)
        kinect_pos = np.array([[xyz_frame[target[1], target[0], 0], xyz_frame[target[1], target[0], 1], xyz_frame[target[1], target[0], 2], 1]])

        xyz = np.dot(calib_mat, kinect_pos.T)
        xyz = xyz[:-1].reshape([1, 3])[0]

        return xyz
