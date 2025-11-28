import Agent
import taichi as ti
from numpy.random import default_rng
import numpy as np
import utils
import AgentConfig
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from gl_utils import Camera
from math_lib import *
from FK import ForwardKinamics
from SkeletonDraw import NativeSkeleton
from LoadBVH import ReadBVHFile
import AgentConfig

def BehaviorSimulation(num = 5):
    JointRadius = 3.0
    SceneWidth = 600
    SceneHeight = 600

    Deltatime = 33

    Gray = np.array([.4627, .5843, .6510], dtype=np.float32)
    Yellow = np.array([.8784, .8706, .7647], dtype=np.float32)
    Red = np.array([1.0, .0, .0], dtype=np.float32)
    LightGreen = np.array([0.0, 1.0, 0.25, 1.0], dtype=np.float32)
    Cyon = np.array([0.4156, 0.6980, 0.6980], dtype=np.float32)
    Pink = np.array([0.7765, 0.0706, 0.9294], dtype=np.float32)
    LightYellow = np.array([0.8784, 0.9412, 0.0235], dtype=np.float32)
    Specular = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    Green = np.array([.0, 1.0, .0], dtype=np.float32)
    Black = np.array([.0, .0, .0], dtype=np.float32)
    White = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    LightGray = np.array([.9490, .9421, .7843], dtype=np.float32)

    colors = [Yellow, LightGreen, Cyon, Pink, LightYellow, Yellow]

    np_axisX = np.array([1,0,0], dtype=np.float32)
    np_axisY = np.array([0,1,0], dtype=np.float32)
    np_axisZ = np.array([0,0,1], dtype=np.float32)

    rng = np.random.default_rng()

    floor_y_pos = -25.
    target_pos = np.array([-1000,-1000,-1000], dtype=np.float32)

    EyePos = np.array([0, 240, 300], dtype=np.float32)
    Center = np.array([0, 0, 100], dtype=np.float32)
    RightVec = np.array([1, 0, 0], dtype=np.float32)

    camera = Camera(SceneWidth, SceneHeight, 60., EyePos, Center, RightVec)

    sk: NativeSkeleton = ReadBVHFile('./resources/b_confid_wlk.bvh')
    sk.RemoveDeepJoint()
    # fkTest.update_skl_data()

    remap_range = [-200, 200, -250, 250]
    N = 100
    crowd_num = num
    curr_t = [rng.integers(0, crowd_num) for _ in range(crowd_num)]
    fks = [ForwardKinamics(sk, True) for _ in range(crowd_num)]
    color_t = [colors[rng.integers(0, 6)] for _ in range(crowd_num)]
    step = 0.01

    current_mode = 2

    flock = Agent.Flock(
        max_n=N,
        pos=rng.random(size=(N, 2), dtype=np.float32),
        vel=np.array([utils.randomvector(2) for _ in range(N)], dtype=np.float32),
        n=min(N,crowd_num),
        obstacle=np.array([[0.2,0.3],[0.4,0.6],[0.8,0.8]]),
        obstacleNum=3,
        step = step,
    ) 

    def ray_plane_intersection(dir):
        p = np.array([0, floor_y_pos, 0], dtype=np.float32)
        return camera.eye + np.dot(p - camera.eye, np_axisY) / np.dot(dir, np_axisY) * dir

    def getTranslationVec(id: int, fk: ForwardKinamics):
        v = fk.global_pos_vec[id]
        return np.array([v[0,3], v[1,3], v[2,3]], dtype=np.float32)

    def remap_from_01(x: float, y: float):
        nx = remap_range[0] + (remap_range[1] - remap_range[0]) / 1 * (x - 0)
        ny = remap_range[2] + (remap_range[3] - remap_range[2]) / 1 * (y - 0)
        return nx, ny

    def remap_to_01(x: float, y: float):
        nx = 0 + 1 / (remap_range[1] - remap_range[0]) * (x - remap_range[0])
        ny = 0 + 1 / (remap_range[3] - remap_range[2]) * (y - remap_range[2])
        return nx, ny

    def clamp(x: float, y: float):
        nx = x
        ny = y
        if x < remap_range[0]:
            nx = remap_range[0]
        if x > remap_range[1]:
            nx = remap_range[1]
        if y < remap_range[2]:
            ny = remap_range[2]
        if y > remap_range[3]:
            ny = remap_range[3]
        return nx, ny

    def drawLimb(start_p, end_p):
        """drawLimb 绘制一个手臂

        Args:
            start_id (int): 开始id
            end_id (int): 结束id
        """
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
        glutSolidSphere(1.6, 20, 20)
        glScalef(1,1,1)
        glPopMatrix()

    def drawJoint(start_p):
        """drawJoint 绘制关节

        Args:
            self_id (int): 自身的id
        """
        glPushMatrix()
        glTranslatef(start_p[0], start_p[1], start_p[2])
        glutSolidSphere(JointRadius, 18, 18)
        glPopMatrix()

    def drawLeader(x, y):
        glPushMatrix()
        glTranslatef(x, 35, y)
        setMaterialPreset(Red)
        glutSolidSphere(4.0, 18, 18)
        glPopMatrix()

    def drawAAgent(agent: ForwardKinamics, pid: int, cid: int):
        cpos = getTranslationVec(cid, agent)
        drawJoint(cpos)
        if pid != -1:
            ppos = getTranslationVec(pid, agent) 
            drawLimb(ppos, cpos)
        for i in range(0, len(sk.joints[cid].child)):
            drawAAgent(agent, cid, sk.joints[cid].child[i])


    def showText(s: str, color, x: float, y: float):
        glPushMatrix()
        glColor3fv(color)
        glDisable(GL_LIGHTING)
        glWindowPos2f(x, y)
        for i in s:
            glutBitmapCharacter(GLUT_BITMAP_8_BY_13, ord(i))
        glEnable(GL_LIGHTING)
        glPopMatrix()

    def displayTexts():
        y = SceneHeight - 20
        x1 = SceneWidth / 2 - 100
        x2 = 20
        showText("Current Mode: " + AgentConfig.METHODS[current_mode], White, x1, y)
        showText('PRESS Q:Seek,W:Flee,E:Arrival,R:Departure,T:Wander,Y:Avoid,U:Seperation', White, x2, y - 20)
        showText('                        I:Alignment,O:Cohension,P:Flocking,SPACE:Leader', White, x2, y - 40)


    def setMaterialPreset(preset):
        """setMaterialPreset 设置材质预设

        Args:
            preset (NdArray): 材质预设类型
        """
        glMaterialfv(GL_FRONT, GL_AMBIENT, preset)
        glMaterialfv(GL_FRONT, GL_DIFFUSE, preset)
        glMaterialfv(GL_FRONT, GL_SPECULAR, Specular)
        glMaterialf(GL_FRONT, GL_SHININESS, 50.0)

    def drawObstacles():
        obs = flock.obstacle.to_numpy()
        for i in obs:
            glPushMatrix()
            nx, ny = remap_from_01(i[0], i[1])
            # print(i[0], i[1], nx, ny)
            # glScalef(1.0, 1.0, (remap_range[3]-remap_range[2])/(remap_range[1]-remap_range[0]))
            glTranslatef(nx, floor_y_pos, ny)
            setMaterialPreset(Green)
            # glutSolidSphere(32.0, 16, 16)
            glScalef(1.0, 1.0, 1.25)
            glutSolidSphere(0.04 * (remap_range[1] - remap_range[0]), 16, 16)
            glPopMatrix()
            # glPushMatrix()
            # glTranslatef(120, 0, 120)
            # setMaterialPreset(Cyon)
            # glutSolidSphere(20, 16, 16)
            # glPopMatrix()

    def drawFloor():
        """drawFloor 绘制地板
        """
        glPushMatrix()
        lx = round(remap_range[0] / 50)
        ly = round(remap_range[2] / 50)
        rx = round(remap_range[1] / 50)
        ry = round(remap_range[3] / 50)
        for i in range(lx, rx):
            for j in range(ly, ry):
                if ((i + j) % 2) == 0:
                    setMaterialPreset(Gray)
                else:
                    setMaterialPreset(LightGray)
                glBegin(GL_QUADS)
                glNormal3f(.0, 1., .0)
                glVertex3f(i * 50.0, floor_y_pos, j * 50.0)
                glVertex3f(i * 50.0 + 50.0, floor_y_pos, j * 50.0)
                glVertex3f(i * 50.0 + 50.0, floor_y_pos, j * 50.0 + 50.0)
                glVertex3f(i * 50.0, floor_y_pos, j * 50.0 + 50.0)
                glEnd()
        glPopMatrix()

    def drawAgents():
        for i in range(len(fks)):    
            glPushMatrix()
            pos = flock.pos[i]
            # vel = flock.vd[i]
            rpx, rpy = remap_from_01(pos.x, pos.y)
            # rvx, rvy = remap_from_01(vel.x, vel.y)
            glTranslatef(rpx, 0, rpy)
            glRotatef(90 - flock.thetad[i] / utils.M_PI * 180, .0, 1.0, .0)
            glScalef(.6, .6, .6)
            curr_t_p = curr_t[i]
            curr_t[i] = (curr_t_p + 1) % sk.frame_count
            fks[i].update_time_to_field(curr_t[i])
            fks[i].update_skl_data()
            setMaterialPreset(color_t[i])
            drawAAgent(fks[i], -1, 0)
            glPopMatrix()
            if current_mode == 10 and i == 0:
                drawLeader(rpx, rpy)
        

    def draw():
        """draw 总体绘制调用函数
        """
        # profiler.start()
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        camera.LookAt()
        drawFloor()
        drawAgents()
        glPushMatrix()
        setMaterialPreset(LightGray)
        glTranslatef(target_pos[0], target_pos[1], target_pos[2])
        glutSolidSphere(6.0, 32, 32)
        glPopMatrix()
        if current_mode == 5:
            drawObstacles()
        displayTexts()
        glFlush()
        # profiler.stop()
        # profiler.print()

    def play_nextframe(_):
        """play_nextframe 播下一帧
        """
        flock.Control()
        flock.Sense(current_mode)
        flock.Act()
        flock.wrap_edge()
        glutPostRedisplay()
        glutTimerFunc(Deltatime, play_nextframe, 1)

    def mouse_func_callback(button, state, x, y):
        nonlocal target_pos
        if button == GLUT_RIGHT_BUTTON and state == GLUT_DOWN:
            dir_np = camera.GenerateViewDir(x,y)
            target_pos = ray_plane_intersection(dir_np)
            rx, ry = remap_to_01(target_pos[0], target_pos[2])
            flock.UpdateTarget(rx, ry)
            glutPostRedisplay()

    def keyboard_func_callback(key, x, y):
        nonlocal current_mode
        if key == b'q' or key == b'Q':
            current_mode = 0
        elif key == b'w' or key == b'W':
            current_mode = 1
        elif key == b'e' or key == b'E':
            current_mode = 2
        elif key == b'r' or key == b'R':
            current_mode = 3
        elif key == b't' or key == b'T':
            current_mode = 4
        elif key == b'y' or key == b'Y':
            current_mode = 5
        elif key == b'u' or key == b'U':
            current_mode = 6
        elif key == b'i' or key == b'I':
            current_mode = 7
        elif key == b'o' or key == b'O':
            current_mode = 8
        elif key == b'p' or key == b'P':
            current_mode = 9
        elif key == b' ':
            current_mode = 10
        glutPostRedisplay()

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
        glutTimerFunc(Deltatime, play_nextframe, 1)
        glutMouseFunc(mouse_func_callback)
        glutKeyboardFunc(keyboard_func_callback)
        glutMainLoop()

    # init()

    def Two_Dui():
        WINDOW_HEIGHT = 500
        AR = 1
        WINDOW_WIDTH = AR * WINDOW_HEIGHT
        N = 100 # people max num
        mode = 2
        step = 0.01

        gui = ti.GUI("BehaviorSimulation", res=(WINDOW_WIDTH, WINDOW_HEIGHT))
        rng = default_rng(seed=42)
        flock = Agent.Flock(
            max_n=N,
            pos=rng.random(size=(N, 2), dtype=np.float32),
            vel=np.array([utils.randomvector(2) for _ in range(N)], dtype=np.float32),
            n=min(N,num),
            obstacle=np.array([[0.2,0.3],[0.4,0.6],[0.8,0.8]]),
            obstacleNum=3,
            step = step,
        )  

        while gui.running:
            # 事件捕捉并更改运行模式
            # change mode
            if gui.get_event((ti.GUI.PRESS)):
                if str.upper(gui.event.key) == 'Q':
                    mode = 0
                elif str.upper(gui.event.key) == 'W':
                    mode = 1
                elif str.upper(gui.event.key) == 'E':
                    mode = 2
                elif str.upper(gui.event.key) == 'R':
                    mode = 3
                elif str.upper(gui.event.key) == 'T':
                    mode = 4
                elif str.upper(gui.event.key) == 'Y':
                    mode = 5
                elif str.upper(gui.event.key) == 'U':
                    mode = 6
                elif str.upper(gui.event.key) == 'I':
                    mode = 7
                elif str.upper(gui.event.key) == 'O':
                    mode = 8
                elif str.upper(gui.event.key) == 'P':
                    mode = 9
                elif gui.event.key == ti.GUI.SPACE:
                    mode = 10
                elif gui.event.key == ti.GUI.RMB:
                    mouse_x, mouse_y = gui.get_cursor_pos()
                    flock.UpdateTarget(mouse_x,mouse_y)
                print(AgentConfig.METHODS[mode])

            flock.Control()
            flock.Sense(mode)
            flock.Act()
            flock.wrap_edge()
            flock.Render(gui,mode,WINDOW_HEIGHT)
            gui.show()
    # Two_Dui()
    init()
