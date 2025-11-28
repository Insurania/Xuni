    
import taichi as ti
import numpy as np
import utils

KDrag = 0.01

@ti.data_oriented
class Rocket:
    def __init__(
        self,
        deltaT = 0.3,
        totalExplosion = 5,
        windSpeed = 0.0,
        MAXROCKETNUM = 100,
    ):
        self.deltaT = deltaT
        self.totalExplosion = totalExplosion
        self.windSpeed = windSpeed
        self.maxRocketNum = MAXROCKETNUM

        # 状态向量
        # state pos:vec2,orientation:vec2, vel:vec2, force:vec2, mass:float
        self.state = ti.Vector.field(n=9, dtype=ti.f32, shape=MAXROCKETNUM)
        self.color = ti.field(ti.i32, shape = MAXROCKETNUM)

        # 状态向量的导数
        # Derivative of the state vector
        self.deriv = ti.Vector.field(n=9, dtype=ti.f32, shape=MAXROCKETNUM)

        '''
        Taichi kernel运行期间无法修改field长度，且field无法擦除元素
        self.mode记录了每个火箭的状态
        0：飞行;1：死亡;-1：爆炸
        如果火箭i的mode[i]=1，位置i可以用于储存新的rocket
        field legth cannot be modified and element cannot be erase during taichi kernel
        this attribute record the living state of every Rocket
        0:flying; 1:dead; -1:explosion
        if dead(mode[i]=1), we can fill this place with new Rocket
        '''
        self.mode = ti.field(ti.i8, shape = MAXROCKETNUM)
        self.mode.fill(1)

        # 记录rocket的爆炸次数
        # Current explosion	
        self.explode = ti.field(ti.i8, shape = MAXROCKETNUM)

        # 存储rocket的三角形顶点
        # rocket is render as triangles
        self.triangle_X = ti.Vector.field(n=2, dtype=ti.f32, shape=MAXROCKETNUM)
        self.triangle_Y = ti.Vector.field(n=2, dtype=ti.f32, shape=MAXROCKETNUM)
        self.triangle_Z = ti.Vector.field(n=2, dtype=ti.f32, shape=MAXROCKETNUM)

    @ti.kernel
    def FindDeadID(self)->ti.i8:
        """
        返回任意一个标记为dead的火箭的 ID，这个位置可以用于储存新的rocket
        return any dead rocket's ID in record
        """
        DeadID = -1
        for i in range (self.maxRocketNum):
            if self.mode[i] == 1:
                DeadID = i
        return DeadID
    
    def InitState(self,value,color):
        """
        设置新火箭的初始状态，必须输入：
            value：numpy，维度为9
            color：int， 如0x00ff00
        Set initial state for a new rocket, must input:
        value:numpy with dim 9
        color:int, eg.0x00ff00
        """
        if value is not None:
            if isinstance(value, np.ndarray):
                newRocket = ti.Vector(value)
                DeadID = self.FindDeadID()
                if DeadID != -1:
                    self.state[DeadID] = newRocket
                    self.color[DeadID] = color
                    self.mode[DeadID] = 0

    @ti.func
    def ComputeForce(self,i):
        """
        ComputeForce()计算施加在火箭 i 上的力
        在这个函数中，你需要为self.state[i][6]和self.state[i][7]设置正确的值，它们是rocket[i]沿X轴和Y轴的受力。
        ComputeForce() computes the forces applied to rocket i
        In this fucntion, you need to set the correct value for self.state[i][6] and self.state[i][7], 
        which are forces along X and Y axis.
        """
        # 你需要实现此函数的内容
        pass

    @ti.func
    def FindDeriv(self,i):
        """
        FindDeriv() 计算火箭 i 的状态向量的导数。
        请记住，火箭的旋转方向与其速度方向对齐。
        在这个函数中，你需要为每个 self.deriv[i] 写入正确的值
        FindDeriv() computes the derivative of the state vector for rocket i.
        Remember that the orientation of the rocket is aligned with it velocity direction.
        In this function you need to write correct value to each self.deriv[i]
        """
        # 你需要实现此函数的内容
        pass
    
    @ti.func
    def UpdateState(self,i):
        """
        UpdateState() 使用火箭 i 的导数向量更新状态向量
        在此函数中，state[i] 应通过 deriv[i] 的值获得正确的值。
        你还应该确定火箭的状态mode[i]。
        如果它达到其最高点，则 mode 为 EXPLOSION（mode = -1）。它为下一个欧拉仿真步骤中的爆炸做准备。
        如果它处于 EXPLOSION 模式，请计算爆炸次数并判断是否已达到 TOTALEXPLOSION 的爆炸次数
        如果已达到，将这个rocket的状态标记为死亡（mode = 1）
        UpdateState() updates the state vector using derivative vector for rocket i
        In this function, state[i] should get correct value using the value from deriv[i].
        You should also determine the mode of the rocket.
        If it reaches its top height, mode is EXPLOSION(mode = -1). It prepares for explosion in the next Euler step of simulation. 
        If it is in EXPLOSION mode, check to see if it has completed TOTALEXPLOSION times of explosion. If so, make it dead.
        Remember to modify explode and dead accordingly. 
        """
        # 你需要实现此函数的内容
        pass

    @ti.kernel
    def EulerStepRocket(self):
        for i in range(self.maxRocketNum):
            if self.mode[i] !=  ti.i8(1): # not dead
                self.ComputeForce(i)
                self.FindDeriv(i)
                self.UpdateState(i)

    @ti.kernel
    def MakeTriangle(self):
        """计算出rocket三角形的顶点"""
        for i in range(self.maxRocketNum):
            triangleSize = 0.02
            pos = ti.Vector([self.state[i][0],self.state[i][1]])
            direction = ti.Vector([self.state[i][2],self.state[i][3]])
            direction,_ = utils.Vec2Normalize(direction)
            X = pos + direction * triangleSize
            self.triangle_X[i] = X
            per_direction = ti.Vector([direction[1],-direction[0]])
            X_back = pos - direction * triangleSize * 1.5
            self.triangle_Y[i] = X_back + per_direction * triangleSize
            self.triangle_Z[i] = X_back - per_direction * triangleSize

    def Render(self,gui):
        self.MakeTriangle()
        # 渲染未死亡的rocket三角形
        # rocket render, only render not dead ones
        for i in range (self.maxRocketNum):
            if self.mode[i] == 0:
                gui.triangle(a=self.triangle_X[i], b=self.triangle_Y[i], c=self.triangle_Z[i],color=self.color[i])
                #gui.circle([self.state[i][0],self.state[i][1]], color=self.color[i], radius=10)

       
        
