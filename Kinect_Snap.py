"""
11.29 version
"""
import PyKinectV2
from PyKinectV2 import *
import PyKinectRuntime
import numpy as np
import cv2
import ctypes
import time
import urx

calib_mat = np.array([[-1.005795, 0.008462, -0.021871, -0.078898], [0.024404 ,1.014035, -0.098330 ,0.032592], [0.046580, -0.041540, -1.057145, 1.101778], [0.0, 0.0, 0.0, 1.0000]])
PI = np.pi
HOME = (90 * PI / 180, -90 * PI / 180, 0, -90 * PI / 180, 0, 0)


class Kinect(object):
    def __init__(self):
        # self.rob = urx.Robot("192.168.10.12")
        self._kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Depth | PyKinectV2.FrameSourceTypes_Infrared)
        self.depth_frame = np.empty([512, 424])
        self.focalLength = 1485.73
        self.centerX = 960.5
        self.centerY = 540.5
        self.lower = np.array([180, 135, 64], dtype="uint8")
        self.upper = np.array([244, 188, 99], dtype="uint8")

    def snap(self):
        while True:
            if self._kinect.has_new_color_frame():
                raw_array = self._kinect.get_last_color_frame()

                # TODO Camera Position confirm.
                raw_img = raw_array.reshape((1080, 1920, 4))                           # to Mat
                flipped_img = cv2.flip(raw_img, 1)                                     # Flipped Image
                cropped_img = flipped_img[108:939, 631:1399]  # cropped ROI, Global View
                result_img = cv2.resize(cropped_img, (256, 256))                       # Resized image (256,256) RGBA
                result_img = cv2.cvtColor(result_img, cv2.COLOR_RGBA2RGB)              # Format : RGB
                result_img = result_img[156:, ]

                mask = cv2.inRange(result_img, self.lower, self.upper)
                output_1 = cv2.bitwise_and(result_img, result_img, mask=mask)
                output = cv2.cvtColor(output_1, cv2.COLOR_RGB2GRAY)

                k = np.argwhere(output != 0).shape[0]

                if k > 5:
                    pass
                    # self.rob.movej(HOME, 1, 1)
                else:
                    break

        return result_img[3:,:,:]

    def color2xyz(self, data):
        """
        :param data: [y,x] pixel list of class index ,dtype = ndarray
        :return: Average position x, y, z
        """

        while True:
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
            pxl_patch[idx][0] = (255 - x) * 3.035 + 573  # Width revision, rate : 3.03515625, offset : 573
            pxl_patch[idx][1] = (y + 128) * 3.1953125 + 143  # Height revision, rate : 3.1953125, offset : 143

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
