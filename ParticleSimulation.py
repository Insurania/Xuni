import taichi as ti
import random
import numpy as np
import utils
import Rocket
import Spark
import math

# config
# 粒子系统参数配置
WINDOW_HEIGHT = 700
AR = 1
WINDOW_WIDTH = int(AR * WINDOW_HEIGHT)
deltaT = 0.01 # 时间步长
MAXROCKETNUM = 10
TOTALEXPLOSION = 5
MAXSPARKNUM = int(MAXROCKETNUM*40*TOTALEXPLOSION)
WINDSPEED = 0.0
MASS = 1
SPARKLIFE = 40.0

# Create rockets storage
rockets = Rocket.Rocket(
    deltaT=deltaT,
    totalExplosion=TOTALEXPLOSION,
    windSpeed=WINDSPEED,
    MAXROCKETNUM=MAXROCKETNUM,
)

# Create sparks storage
sparks = Spark.Spark(
    deltaT=deltaT,
    MAXSPARKNUM=MAXSPARKNUM,
)   


def FireRocket(posx, color):
    """
    当用户按下鼠标左键时调用此函数，在鼠标的X坐标位置处生成一个Rocket并存储在全局变量rockets中
    新生成的Rocket属性限制：只受重力限制，质量为1.
    输入：
        float posx： rocket发射位置的X坐标。rocket发射位置的Y坐标永远设为0
        int color：rocket的HEX颜色，以十六进制表示，为随机数
    FireRocket is called when user presses LEFT MOUSE.
    Input:	float posx. X value of the launch position of the rocket. Y position is always set to 0.
			int color. HEX color of this rocket. It changes according to posx and it is provided for you.
    In this function, you want to generate a Rocket object from Rocket.py and store it in Rockets.
    The initial state of the Rocket object follows the constraints below:
    Position is posx and posy.(posy=0)
    Force is only GRAVITY.
    Mass is 1. 
    """
    pos = utils.vec2(posx,0.0)
    orientation = utils.vec2(random.random()-0.5,1.0)
    vel = utils.ToVelPython(random.randint(3,4),math.atan2(orientation[1],orientation[0]))
    force = utils.vec2(WINDSPEED*deltaT,-utils.GRAVITY)
    newRocket = np.array([pos[0],pos[1],orientation[0],orientation[1],vel[0],vel[1],force[0],force[1],MASS])
    rockets.InitState(newRocket,color)

def Explode(posx,posy,rocketSpeed,color):
    """
    当Rocket到达其最高点时（在EulerStep()函数中判断）调用此函数，它会被调用rocket.totalExplosion次
    此函数在[posx, posy]位置生成一系列的环状spark，并存储在全局变量sparks中
    新生成的spark属性限制：
        spark的数量为随机数，他们的初始位置以中心位置呈环状均匀分布，中心位置为[posx, posy]
        每一个spark的速度大小为随机数，但是速度方向受到发射角度和rocket爆炸时速度的影响
        只受重力限制，质量为存储在全局变量MASS中，spark生命存储在全局变量SPARKLIFE中
    输入：
        float posx：环状spark生成位置的X坐标
        float posy：环状spark生成位置的Y坐标
        float RocketSpeed：rocket在爆炸时X方向的速度
        int color：环状spark的HEX颜色，以十六进制表示，环状spark的颜色与生成它的rocket颜色相同
    Explode is called in EulerStep() when a rocket reaches its top height. 
    It is called totalExplosion times to generate a series of rings of sparks.
    Input: float posx. X position where a ring of sparks are generated.
		   float posy. Y position where a ring of sparks are generated.
           float RocketSpeed. Rocket speed in X direction at the time of explosion.
           int color. HEX color of the rocket. It will also be the color of the sparks it generates.
    In this function, you want to generate a number of sparks which forms a ring at [posx, posy]
    and store them into sparks.
    The initial state vector of each spark follows the constraints below:
    Number of sparks generated is a random number. They evenly distribute on a ring.
    Position of explosion is posx and posy.
    The explosion gives every spark one same random velocity.
            However, the real velocity of each spark is also affected by its shooting angle 
            and the velocity of rocket when it generates the explosion.
    Force on every spark is just the gravity.
    Mass of every spark is in MASS
    Total life of every spark is in SPARKLIFE
    """
    numSparks = random.randint(20,40)
    angle = 360.0/numSparks
    randomVelocity = random.random()
    radius = 0.1
    newSpark = []
    for cnt in range(numSparks):
        pos = utils.vec2(posx+math.cos(cnt*angle*utils.RAD)*radius,posy+math.sin(cnt*angle*utils.RAD)*radius)
        vel = utils.vec2(randomVelocity*math.cos(cnt*angle*utils.RAD)+rocketSpeed,randomVelocity*math.sin(cnt*angle*utils.RAD))
        force = utils.vec2(WINDSPEED*deltaT,-utils.GRAVITY)
        newSpark.append(np.array([pos[0],pos[1],vel[0],vel[1],force[0],force[1],MASS,SPARKLIFE])) 
    sparks.InitState(newSpark,color)

def EulerStep():
    """
    粒子系统的一次整体计算
    One Euler step of genreal simulation.
    """
    # 第一步：迭代访问所有rocket，如果mode状态为-1，即处于爆炸状态，调用函数Explode()生成一系列sparks
    # Step 1. Iterate through every Rocket. Ignore if the rocket is dead.
    #         If the rockets is in explosion mode generate a ring of sparks.
    for i in range(MAXROCKETNUM):
        if rockets.mode[i] == -1: #EXPLOSION
            Explode(rockets.state[i][0],rockets.state[i][1],rockets.state[i][4],rockets.color[i])
    # 第二步：对存活rocket进行模拟计算
    # Step 2. Euler steps for valid rockets.
    rockets.EulerStepRocket()
    # 第三步：对存活spark进行模拟计算
    # Step 3. Euler steps for valid sparks.
    sparks.EulerStepSpark()

def ParticleSimulation():
    gui = ti.GUI("ParticleSimulation", res=(WINDOW_WIDTH, WINDOW_HEIGHT))
    
    while gui.running:
        # 事件捕捉
        if gui.get_event((ti.GUI.PRESS)):
            if gui.event.key == ti.GUI.LMB:
                mouse_x,_ = gui.get_cursor_pos()
                FireRocket(mouse_x,utils.ConvertData(random.random()))

        EulerStep()

        gui.clear(0x000000)
        # label render 渲染提示词
        str = 'PRESS LEFT MOUSE TO FIRE A ROCKET'
        gui.text(content=str, pos=[0,1], font_size=16, color=0xffffff)
        rockets.Render(gui)
        sparks.Render(gui)
        gui.show()
        