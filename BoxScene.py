from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

from gl_utils import Camera
from math_lib import *
from math import *

import numpy as np
import taichi as ti

def box_scene(step: int):
    """box_scene 生成box场景的主函数

    """    
    # 基本定义常量
    SceneWidth = 600
    SceneHeight = 600

    Gray = np.array([.4627, .5843, .6510], dtype=np.float32)
    Yellow = np.array([.8784, .8706, .7647], dtype=np.float32)
    LightGreen = np.array([0.0, 1.0, 0.25, 1.0], dtype=np.float32)
    Cyon = np.array([0.4156, 0.6980, 0.6980], dtype=np.float32)
    Pink = np.array([0.7765, 0.0706, 0.9294], dtype=np.float32)
    LightYellow = np.array([0.8784, 0.9412, 0.0235], dtype=np.float32)
    Specular = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

    EyePos = np.array([5, 5, 10], dtype=np.float32)
    Center = np.array([5, 2, 0], dtype=np.float32)
    RightVec = np.array([1, 0, 0], dtype=np.float32)

    camera = Camera(SceneWidth, SceneHeight, 60., EyePos, Center, RightVec)

    @ti.kernel
    def mul_mat4(a: mat4, b: mat4) -> mat4:
        """mul_mat4 taichi矩阵乘法类

        Args:
            a (mat4): 矩阵a
            b (mat4): 矩阵b

        Returns:
            mat4: a和b的矩阵乘法结果
        """
        return a @ b
    
    @ti.kernel
    def inv_mat4(a: mat4) -> mat4:
        """inv_mat4 taichi矩阵求逆类

        Args:
            a (mat4): 矩阵a

        Returns:
            mat4: inv(a)
        """
        return a.inverse()

    """Write Your Matrix Here!!!

    你需要做的工作：把需要的框填到线框里
        B1: GREEN
        B2: PINK
        B3: CYON
        B2移动后的位置: YELLOW

    你需要完成下面的三个Task：

    Task 1:
        填写 B1_world, B2_world, B3_world，这样可以在对应位置绘制一个box线框

    Task 2:
        填写 B1_local, B2_local, B3_local，它们分别是相对于世界坐标系、B1局部坐标系、B2局部坐标系的变换矩阵

    Task 3:   
        填写 B2TransformMatrix，它是在B2局部坐标系下的变换矩阵，将B2移动到对应的位置

    """

    B1_World = eular_translation_to_transmat(0, 0, 0, 3, 2, 1)
    B2_World = eular_translation_to_transmat(0, 0, 45, 6, 0, 0)
    B3_World = eular_translation_to_transmat(0, 90, 0, 8, 5*math.sqrt(2)/2, 0)

    B1_Local = B1_World
    B2_Local = mul_mat4(inv_mat4(B1_World), B2_World)
    B3_Local = mul_mat4(inv_mat4(B2_World), B3_World)

    Target_B2_World = mat4([[1, 0, 0, 3], [0, 1, 0, 4], 
                            [0, 0, 1, 1], [0, 0, 0, 1]])

    B2TransformMatrix = mul_mat4(inv_mat4(B2_World), Target_B2_World)




    

    # B1_World = mat4([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    # B2_World = mat4([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    # B3_World = mat4([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


    # B1_Local = mat4([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    # B2_Local = mat4([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    # B3_Local = mat4([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    # Target_B2_World = mat4([[1, 0, 0, 3], [0, 1, 0, 4], 
    #                         [0, 0, 1, 1], [0, 0, 0, 1]])

    # B2TransformMatrix = mat4([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])



    def setMaterialPreset(preset):
        """setMaterialPreset 设置材质预设

        Args:
            preset (NdArray): 材质预设类型
        """
        glMaterialfv(GL_FRONT, GL_AMBIENT, preset)
        glMaterialfv(GL_FRONT, GL_DIFFUSE, preset)
        glMaterialfv(GL_FRONT, GL_SPECULAR, Specular)
        glMaterialf(GL_FRONT, GL_SHININESS, 50.0)

    def drawFloor():
        """drawFloor 绘制地板
        """
        glBegin(GL_QUADS)
        setMaterialPreset(Gray)
        glNormal3f(.0, 1., .0)
        glVertex3f(-.01, -.01, -1.01)
        glVertex3f(10.01, -.01, -1.01)
        glVertex3f(10.01, -.01, 5.01)
        glVertex3f(-0.01, -.01, 5.01)
        glEnd()

        glBegin(GL_QUADS)
        setMaterialPreset(Yellow)
        glNormal3f(.0, .0, 1.0)
        glVertex3f(-.01, -.01, -1.01)
        glVertex3f(10.01, -.01, -1.01)
        glVertex3f(10.01, 8.01, -1.01)
        glVertex3f(-0.01, 8.01, -1.01)
        glEnd()

    def apply_mat4(m: mat4):
        """apply_mat4 应用矩阵

        Args:
            m (mat4): 执行移动和旋转
        """
        glTranslatef(m[0,3], m[1,3], m[2,3])
        axis = get_axis_angle_from_matrix4x4(m)
        glRotatef(axis.w * 180 / math.pi, axis.x, axis.y, axis.z)

    def drawRectangle(useWireFrame: bool):
        """drawRectangle 绘制立方体

        Args:
            useWireFrame (bool): 是否使用线框模式
        """
        if useWireFrame:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glBegin(GL_QUADS)
        glNormal3f(.0, .0, 1.0)
        glVertex3f(.0, .0, .0)
        glVertex3f(3.0, .0, .0)
        glVertex3f(3.0, 2.0, .0)
        glVertex3f(.0, 2.0, .0)

        glNormal3f(-1., .0, .0)
        glVertex3f(.0, .0, .0)
        glVertex3f(.0, .0, -1.0)
        glVertex3f(.0, 2.0, -1.0)
        glVertex3f(.0, 2.0, .0)

        glNormal3f(.0, -1., .0)
        glVertex3f(.0, .0, .0)
        glVertex3f(3.0, .0, .0)
        glVertex3f(3.0, .0, -1.0)
        glVertex3f(.0, .0, -1.0)

        glNormal3f(1., .0, .0)
        glVertex3f(3.0, 2.0, -1.0)
        glVertex3f(3.0, .0, -1.0)
        glVertex3f(3.0, .0, .0)
        glVertex3f(3.0, 2.0, .0)

        glNormal3f(.0, .0, -1.)
        glVertex3f(3.0, 2.0, -1.0)
        glVertex3f(3.0, .0, -1.0)
        glVertex3f(.0, .0, -1.0)
        glVertex3f(.0, 2.0, -1.0)

        glNormal3f(.0, 1., .0)
        glVertex3f(3.0, 2.0, -1.0)
        glVertex3f(.0, 2.0, -1.0)
        glVertex3f(.0, 2.0, .0)
        glVertex3f(3.0, 2.0, .0)

        glEnd()

    def draw_world_wireframe():
        """draw_world_wireframe 根据世界坐标绘制线框立方体
        """
        glPushMatrix()
        apply_mat4(B1_World)
        setMaterialPreset(LightGreen)
        drawRectangle(True)
        glPopMatrix()
        glPushMatrix()
        apply_mat4(B2_World)
        setMaterialPreset(Pink)
        drawRectangle(True)
        glPopMatrix()
        glPushMatrix()
        apply_mat4(B3_World)
        setMaterialPreset(Cyon)
        drawRectangle(True)
        glPopMatrix()
        
        
    def draw_local_filled():
        """draw_local_filled 根据局部坐标绘制填充立方体
        """
        glPushMatrix()
        apply_mat4(B1_Local)
        setMaterialPreset(LightGreen)
        drawRectangle(False)
        glPushMatrix()
        apply_mat4( B2_Local)
        setMaterialPreset(Pink)
        drawRectangle(False)
        glPushMatrix()
        apply_mat4(B3_Local)
        setMaterialPreset(Cyon)
        drawRectangle(False)
        glPopMatrix() 
        glPopMatrix() 
        glPopMatrix()    

    def draw_trans_filled():
        """draw_trans_filled 绘制移动的线框和填充立方体
        """
        glPushMatrix()
        apply_mat4(Target_B2_World)
        setMaterialPreset(LightYellow)
        drawRectangle(True)
        glPopMatrix()    
        glPushMatrix()
        apply_mat4(B1_Local)
        glPushMatrix()
        apply_mat4(B2_Local)
        glPushMatrix()
        apply_mat4(B2TransformMatrix)
        setMaterialPreset(LightYellow)
        drawRectangle(False)
        glPopMatrix()
        glPopMatrix()
        glPopMatrix() 

    def draw():
        """draw 总体绘制调用函数
        """
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        camera.LookAt()
        drawFloor()
        if step == 1:
            draw_world_wireframe()
        if step > 1:
            draw_local_filled()
        if step > 2:
            draw_trans_filled()
        # glPushMatrix()
        # glTranslatef(.0, .0, 1.0)
        # setMaterialPreset(LightGreen)
        # drawRectangle(True)
        # glPopMatrix()
        glFlush()

    def init():
        """init 初始化函数
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
        glutDisplayFunc(draw)
        glutMainLoop()

    init()