from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from OpenGL.arrays import *

import taichi as ti
import numpy as np

from main import RunType
from LoadBVH import ReadBVHFile
from NativeSkeleton import NativeSkeleton
from FK import ForwardKinamics
from Blend import AnimationBlender, BlendMethod
from IK import InverseKinamics
from gl_utils import Camera
from math_lib import vec3, ray_sphere_list_intersection

def draw_animation(file_name: str, play_type = RunType.FK, start_frame = -1, file_name2: str = "", start_frame2 = 1, blend_frame_count = 10, blend_method = "SLERP"):
    """draw_animation 绘制动画的核心函数

    Args:
        file_name (str): 文件名
        use_fk (bool, optional): 是否使用fk. Defaults to True.
        start_frame (int, optional): IK解算的开始帧. Defaults to -1. 
    """

    # 初始化骨骼
    sk: NativeSkeleton = ReadBVHFile(file_name) 
    fk: ForwardKinamics = ForwardKinamics(sk)
    ik: InverseKinamics = None
    blend: AnimationBlender = None
    if play_type == RunType.FK:
        fk.update_time(0)
    elif play_type == RunType.BLEND:
        sk2 = ReadBVHFile(file_name2)
        fk2 = ForwardKinamics(sk2)
        if blend_method == "SLERP":
            blend = AnimationBlender(fk, fk2, start_frame, start_frame2, blend_frame_count, BlendMethod.SLERP)
        elif blend_method == "SQUAD":
            blend = AnimationBlender(fk, fk2, start_frame, start_frame2, blend_frame_count, BlendMethod.SQUAD)
        elif blend_method == "CUBIC":
            blend = AnimationBlender(fk, fk2, start_frame, start_frame2, blend_frame_count, BlendMethod.CUBIC)
        else: 
            raise Exception("Undefined blend method : " + blend_method)
        
    else:
        fk.init_state(start_frame)
        ik = InverseKinamics(fk)

    in_ik_sol = False

    frame_cnt = 0

    floor_y_pos = fk.local_matrix[0][1,3] - 40.

    # https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyopengl
    # 颜色定义
    Gray = np.array([.188, .2705, .2863], dtype=np.float32)
    LightGray = np.array([.9490, .9421, .7843], dtype=np.float32)
    DarkYellow = np.array([0.9922, 0.8392, 0.1725, 1.0], dtype=np.float32)
    Blue = np.array([0.1233, 0.2292, 0.9725, 1.0], dtype=np.float32)
    Cyon = np.array([0.4156, 0.6980, 0.6980], dtype=np.float32)
    Pink = np.array([0.7765, 0.0706, 0.9294], dtype=np.float32)
    LightRed = np.array([1.0, 0.2, 0.2, 1.0])
    LightGreen = np.array([0.0, 1.0, 0.25, 1.0], dtype=np.float32)
    Specular = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

    # 常量定义
    JointRadius = 2.0
    LimbScale = 2.0
    CoordLength = 4.0
    Deltatime = 33

    SceneWidth = 600
    SceneHeight = 600

    # 相机设置
    FOV = 60.
    EyePos = np.array([0, 80, 200], dtype=np.float32)
    Center = np.array([.0, .0, .0], dtype=np.float32)
    CameraRight = np.array([1.0, .0, .0], dtype=np.float32)

    camera = Camera(SceneWidth, SceneHeight, FOV, EyePos, Center, CameraRight)
    
    # 计算用设立
    np_axisX = np.array([1,0,0], dtype=np.float32)
    np_axisY = np.array([0,1,0], dtype=np.float32)
    np_axisZ = np.array([0,0,1], dtype=np.float32)

    select_joint_id = -1
    select_ori_pos = np.array([0,0,0], dtype=np.float32)
    curr_mouse_posX = 0
    curr_mouse_posY = 0

    def drawFloor():
        """drawFloor 绘制地板
        """
        glPushMatrix()
        for i in range(-10, 10):
            for j in range(-10, 15):
                if ((i + j) % 2) == 0:
                    setMaterialPreset(Gray)
                else:
                    setMaterialPreset(LightGray)
                glBegin(GL_QUADS)
                glNormal3f(.0, 1., .0)
                glVertex3f(i * 10.0, floor_y_pos, j * 10.0)
                glVertex3f(i * 10.0 + 10.0, floor_y_pos, j * 10.0)
                glVertex3f(i * 10.0 + 10.0, floor_y_pos, j * 10.0 + 10.0)
                glVertex3f(i * 10.0, floor_y_pos, j * 10.0 + 10.0)
                glEnd()
        glPopMatrix()

    def draw():
        """draw 核心绘制函数
        """
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        camera.LookAt()
        glPushMatrix()
        drawFloor()
        # glRotatef(-90, 0, 0, 1)
        # glTranslatef(0, 0, 20)
        # glMaterialfv(GL_FRONT, GL_DIFFUSE, LightRed)
        # glutSolidSphere(10.0, 32, 32)
        if in_ik_sol and select_joint_id > 0:
            dist = np.dot(camera.front, GetTranslationVec(select_joint_id) - camera.eye)
            # print(GetTranslationVec(select_joint_id), dist, camera.front)
            target_pos = camera.GetHitPosition(curr_mouse_posX, curr_mouse_posY, dist)
            # print(target_pos)
            glPushMatrix()
            glTranslatef(target_pos[0], target_pos[1], target_pos[2])
            setMaterialPreset(Blue)
            glutSolidSphere(1, 48, 48)
            glPopMatrix()
            ik.set_target_position(target_pos)
        drawRecursive(-1, 0)
        glPopMatrix()
        glFlush()

    def GetTranslationVec(id):
        """GetTranslationVec 获取变换矩阵

        Args:
            id (int): 在fk链中的下标

        Returns:
            vec3: 位置坐标
        """
        if play_type != RunType.BLEND:
            mat = fk.global_matrix[id]
        else:
            mat = blend.get_current_matrix(id)
        return np.array([mat[0,3], mat[1,3], mat[2,3]], dtype=np.float32)

    def drawLimb(start_id, end_id):
        """drawLimb 绘制一个手臂

        Args:
            start_id (int): 开始id
            end_id (int): 结束id
        """
        start_p = GetTranslationVec(start_id)
        end_p = GetTranslationVec(end_id)
        l = np.linalg.norm(end_p - start_p)
        dir = (end_p - start_p) / l
        dx = dir[1] + dir[2]
        dy = dir[0] + dir[2]
        dz = dir[0] + dir[1]
        axis_t = 0
        axis = None
        angle = 0
        if dx >= dy and dx >= dz:
            axis_t = 0
            axis = np.cross(np_axisX, dir)
            angle = np.arccos(np.dot(dir, np_axisX)) * 180 / np.pi
        elif dy >= dx and dy >= dz:
            axis_t = 1
            axis = np.cross(np_axisY, dir)
            angle = np.arccos(np.dot(dir, np_axisY)) * 180 / np.pi
        else:
            axis_t = 2
            axis = np.cross(np_axisZ, dir)
            angle = np.arccos(np.dot(dir, np_axisZ)) * 180 / np.pi
        
        glPushMatrix()
        center = (start_p + end_p) / 2
        glTranslatef(center[0], center[1], center[2])
        glRotatef(angle, axis[0], axis[1], axis[2])
        if axis_t == 0:
            glScalef(l / 2, 1, 1)
        elif axis_t == 1:
            glScalef(1, l / 2, 1)
        else:
            glScalef(1, 1, l / 2)
        if play_type == RunType.IK and (ik.is_in_ik_chain(end_id) or ik.is_in_ik_chain(start_id)):
            setMaterialPreset(Cyon)
        else:
            setMaterialPreset(DarkYellow)
        glutSolidSphere(1, 48, 48)
        glScalef(1,1,1)
        glPopMatrix()

    def setMaterialPreset(preset):
        """setMaterialPreset 获取材质

        Args:
            preset (NdArray): 材质类型
        """
        glMaterialfv(GL_FRONT, GL_AMBIENT, preset)
        glMaterialfv(GL_FRONT, GL_DIFFUSE, preset)
        glMaterialfv(GL_FRONT, GL_SPECULAR, Specular)
        glMaterialf(GL_FRONT, GL_SHININESS, 50.0)

    def drawJoint(self_id):
        """drawJoint 绘制关节

        Args:
            self_id (int): 自身的id
        """
        glPushMatrix()
        start_p = GetTranslationVec(self_id)
        glTranslatef(start_p[0], start_p[1], start_p[2])
        if self_id == select_joint_id:
            setMaterialPreset(LightRed)
        elif play_type == RunType.IK and ik.is_in_ik_chain(self_id):
            setMaterialPreset(Pink)
        else:
            setMaterialPreset(LightGreen)
        glutSolidSphere(JointRadius, 18, 18)
        glPopMatrix()

    def drawRecursive(parent_id: int, self_id: int):
        """drawRecursive 递归绘制

        Args:
            parent_id (int): 父id
            self_id (int): 自己id
        """
        drawJoint(self_id)
        if parent_id != -1:
            drawLimb(parent_id, self_id)
        for i in range(0, len(sk.joints[self_id].child)):
            drawRecursive(self_id, sk.joints[self_id].child[i])

    def init():
        """init 初始化函数

        Args:
            play_anim (bool): 是否播放动画
            solve_ik (bool, optional): 是否解算函数
        """
        glutInit()
        glutInitDisplayMode(GLUT_SINGLE | GLUT_RGBA)
        glutInitWindowSize(SceneWidth, SceneHeight)
        glutCreateWindow("Test")
        glClearColor(.0, .0, .0, 1.0)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_NORMALIZE)
        glEnable(GL_BLEND)
        glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST)
        light_position = np.array([200.0, 200.0, 200.0, 0.0], dtype=np.float32)
        glLightfv(GL_LIGHT0, GL_POSITION, light_position)
        light_col = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        amb_col = np.array([.3, .3, .3, .3], dtype=np.float32)
        glLightfv(GL_LIGHT0, GL_AMBIENT, amb_col)
        glLightfv(GL_LIGHT0, GL_DIFFUSE, light_col)
        glLightfv(GL_LIGHT0, GL_SPECULAR, light_col)
        nonlocal play_type
        if (play_type == RunType.FK) or (play_type == RunType.BLEND):
            glutTimerFunc(Deltatime, play_nextframe, 1)
        if play_type == RunType.IK:
            glutMouseFunc(mouse_func_callback)
            glutMotionFunc(mouse_motion_func_callback)
        glutDisplayFunc(draw)
        glutMainLoop()

    def play_nextframe(_):
        """play_nextframe 播下一帧
        """
        if play_type == RunType.FK:
            fk.frame_cnt = (fk.frame_cnt + 1) % sk.frame_count
            fk.update_time(fk.frame_cnt)
        else:
            nonlocal frame_cnt
            frame_cnt = frame_cnt + 1
            blend.update_timestep(frame_cnt)
        glutPostRedisplay()
        glutTimerFunc(Deltatime, play_nextframe, 1)

    def mouse_func_callback(button, state, x, y):
        """mouse_func_callback 鼠标事件响应

        Args:
            button (int): 鼠标按钮
            state (int): 按下状态
            x (float): 屏幕坐标
            y (float): 屏幕坐标
        """
        nonlocal in_ik_sol, select_joint_id, select_ori_pos
        if button == GLUT_RIGHT_BUTTON and state == GLUT_DOWN:
            dir_np = camera.GenerateViewDir(x,y)
            dir = vec3(dir_np[0], dir_np[1], dir_np[2])
            select_joint_id = ray_sphere_list_intersection(camera.eye, dir, JointRadius, fk.global_position)
            if id != -1:
                select_ori_pos = GetTranslationVec(select_joint_id)
            ik.set_select_joint_id(select_joint_id)
            glutPostRedisplay()
            
        elif button == GLUT_LEFT_BUTTON and state == GLUT_DOWN:
            in_ik_sol = True
            # curr_mouse_posX = x
            # curr_mouse_posY = y
            # glutPostRedisplay()
        elif button == GLUT_LEFT_BUTTON and state == GLUT_UP:
            in_ik_sol = False
    
    def mouse_motion_func_callback(x, y):
        """mouse_motion_func_callback 鼠标移动函数

        Args:
            x (float): 鼠标x坐标
            y (float): 鼠标y坐标
        """
        nonlocal curr_mouse_posX, curr_mouse_posY
        curr_mouse_posX = x
        curr_mouse_posY = y
        glutPostRedisplay()

    init()