import taichi as ti
import numpy as np
import utils

KDrag = 0.01
# Coefficients of restitution
COR = 0.4

@ti.data_oriented
class Spark:
    def __init__(
        self,
        deltaT = 0.3,
        MAXSPARKNUM = 1000,
    ):
        self.deltaT = deltaT
        self.maxSparkNum = MAXSPARKNUM

        # 状态向量
        # state pos:vec2, vel:vec2, force:vec2, mass:float, life:float
        self.state = ti.Vector.field(n=8, dtype=ti.f32, shape=MAXSPARKNUM)
        self.color = ti.field(ti.i32, shape = MAXSPARKNUM)

        # 状态向量的导数
        # Derivative of the state vector
        self.deriv = ti.Vector.field(n=8, dtype=ti.f32, shape=MAXSPARKNUM)

        '''
        Taichi kernel运行期间无法修改field长度，且field无法擦除元素 
        self.mode记录了每个spark的状态
        0：飞行;1：死亡
        如果spark i的mode[i]=1，位置i可以用于储存新的spark
        field legth cannot be modified and element cannot be erase during taichi kernel
        this attribute record the living state of every Spark
        0:flying; 1:dead
        if dead(mode[i]=1), we can fill this place with new Spark
        '''
        self.mode = ti.field(ti.i8, shape = MAXSPARKNUM)
        self.mode.fill(1)

    def FindDeadID(self):
        """return any dead spark's ID in record"""
        DeadID = -1
        for i in range (self.maxSparkNum):
            if self.mode[i] == 1:
                DeadID = i
                break
        return DeadID
    
    def InitState(self,value,color):
        """
        设置新火箭的初始状态，必须输入：
            value：列表，元素是维度为8的numpy数组
            color：int， 如0x00ff00
        Set initial state for a new spark, must input:
        value:list of a series of numpy with dim 8
        color:int, eg.0x00ff00
        """
        DeadID = self.FindDeadID()
        if DeadID >= 0:
            cnt = DeadID
            for spark in value:
                newSpark = ti.Vector(spark)
                self.state[cnt] = newSpark
                self.color[cnt] = color
                self.mode[cnt] = 0
                cnt+=1

    @ti.func
    def ComputeForce(self,i):
        """
        ComputeForce()计算施加在spark i 上的力
        在这个函数中，你需要为 self.state[i][4] 和 self.state[i][5] 设置正确的值，它们是spark[i]沿X轴和Y轴的受力。
        ComputeForce() computes the forces applied to spark i
        In this fucntion, you need to set the correct value for self.state[i][4] and self.state[i][5], 
        which are forces along X and Y axis.
        """
        # 你需要实现此函数的内容
        pass

    @ti.func
    def FindDeriv(self,i):
        """
        FindDeriv()计算 spark i 的状态向量的导数。
        在这个函数中，你需要为每个 self.deriv[i] 写入正确的值
        记住为spark减去它的寿命。
        FindDeriv() computes the derivative of the state vector for spark i.
        In this function you need to write correct value to each self.deriv[i]
        Remember to substract its life counts.
        """
        # 你需要实现此函数的内容
        pass

    @ti.func
    def UpdateState(self,i):
        """
        UpdateState() 使用 spark i 的导数向量更新状态向量
        在此函数中，state[i] 应使用 deriv[i] 的值获得正确的值。
        你应该考虑spark反弹到地面上的情况，使用给定的 COR（恢复系数）对spark的弹跳进行建模。
        请记住，当其生命计数低于 0 时将其设置为死亡（self.mode[i] 应设置为 1）
        UpdateState() updates the state vector using derivative vector for spark i
        In this function, state[i] should get correct value using the value from deriv[i].
        You should consider the situation when the spark bounce onto the ground. 
        You should model its boucing using the given COR(coefficients of restitution).
        Remember to set it to dead when its life count falls below 0;
        When spark i is dead, self.mode[i] should be set to 1
        """
        # 你需要实现此函数的内容
        pass

    @ti.kernel
    def EulerStepSpark(self):
        for i in range(self.maxSparkNum):
            if self.mode[i] != 1: # not dead
                self.ComputeForce(i)
                self.FindDeriv(i)
                self.UpdateState(i)

    def Render(self,gui):
        # 渲染存活的spark
        # spark render, only render not dead ones
        for i in range (self.maxSparkNum):
            if self.mode[i] == 0:
                gui.circle([self.state[i][0],self.state[i][1]], color=self.color[i], radius=4)