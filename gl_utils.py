"""本文件包含了一些用在图形库里的类

Author:
    lire 

Date:
    2025-11-02

Version:
    1.0
"""
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from OpenGL.arrays import *

from threading import Timer

import math
import numpy as np

class Camera:

    """Camera class for displaying

    Attributes:
        fov: field of view
        nearP: distance to near plane
        farP: distance to far plane
        eye: position of eye
        center: position of center of scene
        front: front vector of camera
        right: right vector of camera
        up: up vector of camera
    """
    def __init__(self, wid, hei, fov, eye, center, right) -> None:
        self.fov = fov
        self.nearP = .1
        self.farP = 1000.0
        self.eye = eye
        self.center = center
        self.front = self.center - self.eye
        self.dist = np.linalg.norm(self.front)
        self.front = self.front / self.dist
        self.right = right
        self.up = np.cross(self.right, self.front)
        self.width = wid
        self.height = hei
        

    def LookAt(self):
        """Generate Projection Matrix

        We use perspective projection to generate the scene.
        And we should call gluLookAt for camera position, 
        which is used for Model-View transformation.

        """
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(self.fov, 1, self.nearP, self.farP)
        glMatrixMode(GL_MODELVIEW)
        gluLookAt(self.eye[0], self.eye[1], self.eye[2],
                  self.center[0], self.center[1], self.center[2],
                  self.up[0], self.up[1], self.up[2])
        
    def GenerateViewDir(self, x: float, y: float):
        """Generate the view dir in the near plane

        Args:
            x (float): x of coordinate in window
            y (float): y of coordinate in window

        Returns:
            Vector3: direction of view
        """
        fx = 2 * (x / self.width) - 1
        fy = 1 - 2 * (y / self.height)
        max_ver_d_half = self.nearP * math.tan(self.fov / 360 * math.pi)
        max_hor_d_half = max_ver_d_half * self.width / self.height
        dir = self.front * self.nearP + fx * max_hor_d_half * self.right + fy * max_ver_d_half * self.up
        return dir / np.linalg.norm(dir)
    
    def GetHitPosition(self, x: float, y: float, dist_plane: float):
        """Generate the hit position in the plane with distance of dist_plane to camera

        Args:
            x (float): x of coordinate in window
            y (float): y of coordinate in window

        Returns:
            Vector3: position of view in world space
        """
        fx = 2 * (x / self.width) - 1
        fy = 1 - 2 * (y / self.height)
        max_ver_d_half = dist_plane * math.tan(self.fov / 360 * math.pi)
        max_hor_d_half = max_ver_d_half * self.width / self.height
        dir = self.front * dist_plane + fx * max_hor_d_half * self.right + fy * max_ver_d_half * self.up
        return self.eye + dir
