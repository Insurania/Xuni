import taichi as ti
import numpy as np
import utils
import AgentConfig

@ti.data_oriented
class Flock:
    def __init__(
        self,
        max_n,
        pos=None,
        vel=None,
        acc=None,
        obstacleNum = None,
        obstacle = None,
        n = None,
        step = None,
    ):

        self.n = n
        self.step = step
        self.pos = ti.Vector.field(n=2, dtype=ti.f32, shape=max_n)
        self.vel = ti.Vector.field(n=2, dtype=ti.f32, shape=max_n)
        self.acc = ti.Vector.field(n=2, dtype=ti.f32, shape=max_n)

        self.target = ti.field(ti.f32,shape=(2,))
        self.target.fill(0.5)
        self.obstacle = ti.Vector.field(n=2,dtype = ti.f32,shape=obstacleNum)
        self.obstacleNum = obstacleNum

        self.InitField(self.pos, pos)
        self.InitField(self.vel, vel)
        self.InitField(self.acc, acc)
        self.InitField(self.obstacle,obstacle)

        self.wander_upd = 0

        # Input vector dimension: 2 
        # 0 : force;1 : torque
        self.input = ti.Vector.field(n=2, dtype=ti.f32, shape=max_n)
        # State vector dimension: 4
        # 0 : position in local coordinates. Useless.
        # 1 : orientation angle in global coordinates.
        # 2 : velocity in local coordinates.
        # 3 : angular velocity in global coordinates.
        self.state = ti.Vector.field(n=4, dtype=ti.f32, shape=max_n)
        
        # Needed in Wander behavior
        # Wander velocity
        self.vWander = ti.Vector.field(n=2, dtype=ti.f32, shape=max_n)
        # Nominal velocity
        self.v0 = ti.Vector.field(n=2, dtype=ti.f32, shape=max_n)
        self.InitState(max_n)
        
        # Derivative vector
        self.deriv = ti.Vector.field(n=4, dtype=ti.f32, shape=max_n)
        
        # Control inputs: Desired velocity
        self.vd = ti.field(ti.f32,shape=(max_n,))

        # Desired orientation
        self.thetad = ti.field(ti.f32,shape=(max_n,))

    def InitField(self, field, value):
        if value is not None:
            if isinstance(value, np.ndarray):
                field.from_numpy(value)
            else:
                field.from_numpy(
                    np.full(fill_value=value, dtype=np.float32, shape=self.max_n))
                
    @ti.kernel
    def InitState(self,n:ti.i32):
        for i in range(n):
            angle = (ti.random(float) % 360 - 180) / 180.0 *utils.M_PI
            self.state[i][1] = angle
            self.vWander[i][0] = ti.cos(angle) * AgentConfig.KWander
            self.vWander[i][1] = ti.sin(angle) * AgentConfig.KWander
            self.v0[i][0] = ti.cos(angle) * AgentConfig.MaxVelocity / 2.0
            self.v0[i][1] = ti.sin(angle) * AgentConfig.MaxVelocity / 2.0

    @ti.func
    def LocalToWorld(self,i,vecl):
        # i: agent id; vec: ti.Vector(2); return : ti.Vector(2)
        s = ti.sin(self.state[i][1])
        c = ti.cos(self.state[i][1])
        w = ti.Vector([c*vecl[0]-s*vecl[1],s*vecl[0]+c*vecl[1]])
        return w
    
    @ti.func
    def WorldToLocal(self,i,vecw):
        # i: agent id; vec: ti.Vector(2); return : ti.Vector(2)
        s = ti.sin(self.state[i][1])
        c = ti.cos(self.state[i][1])
        l = ti.Vector([c*vecw[0]+s*vecw[1],-s*vecw[0]+c*vecw[1]])
        return l

    @ti.kernel
    def Control(self):
        """
        在给定所需速度 vd 和所需方向 thetad 的情况下应用控制规则。
        速度控制：input[0] = f = m *[Kv0 * (vd-state[2])]
        航向控制：input[1] = tau = I * [-Kv1 * state[3] + Kp1 * (state[1]-thetad)]
        此函数应当为所有agent适当地设置 input[0] 和 input[1]
        You should apply the control rules given desired velocity vd and desired orientation thetad.
        Velocity control: input[0] = f = m *[Kv0 * (vd-state[2])]
        Heading control: input[1] = tau = I * [-Kv1 * state[3] + Kp1 * (state[1]-thetad)]
        This function sets input[0] and input[1] appropriately after being called for all agent
        """
        for i in range (self.n):
            self.input[i][0] = AgentConfig.Mass*AgentConfig.Kv0*(self.vd[i]-self.state[i][2])
            self.input[i][0] = utils.Truncate(self.input[i][0],-AgentConfig.MaxForce,AgentConfig.MaxForce)
            self.input[i][1] = AgentConfig.Inertia*(-AgentConfig.Kv1 * self.state[i][3] + AgentConfig.Kp1*(self.state[i][1]-self.thetad[i]))
            self.input[i][1] = utils.Truncate(self.input[i][1],-AgentConfig.MaxTorque,AgentConfig.MaxTorque)

    @ti.func
    def FindDeriv(self,i):
        """
        计算给定agent i 的self.input[i]和self.state[i]的导数向量self.deriv[i]
        Compute derivative vector given input and state vectors for agent i
        This function sets derive vector to appropriate values after being called
        """
        self.deriv[i][3] = self.input[i][1]/AgentConfig.Inertia 
        self.deriv[i][2] = (self.input[i][0]/AgentConfig.Mass) 
        self.deriv[i][0] = self.deriv[i][2] * self.step
        self.deriv[i][1] = self.deriv[i][3] * self.step

    @ti.func
    def UpdateState(self,i):
        """
        通过agent i的导数向量更新其状态向量
        通过utils.Truncate()执行验证检查以确保所有值都在 MAX 值范围内
        Update the state vector given derivative vector for agent i
        Perform validation check to make sure all values are within MAX values
        """
        self.state[i] = self.deriv[i]
        self.state[i][2] = utils.Truncate(self.state[i][2],-AgentConfig.MaxVelocity,AgentConfig.MaxVelocity)
        self.state[i][3] = utils.Truncate(self.state[i][3],-AgentConfig.MaxAngVel*utils.M_PI/180,AgentConfig.MaxAngVel*utils.M_PI/180)

    @ti.kernel
    def wrap_edge(self):
        for i in range(self.n):
            if self.pos[i][0] < 0:
                self.pos[i][0] = 0
            if self.pos[i][0] > 1:
                self.pos[i][0] = 1
            if self.pos[i][1] < 0:
                self.pos[i][1] = 0
            if self.pos[i][1] > 1:
                self.pos[i][1] = 1

    @ti.kernel
    def Act(self):
        for i in range(self.n):
            self.FindDeriv(i)
            self.UpdateState(i) 
            gVelocity = utils.ToVel(self.vd[i],self.thetad[i])
            self.pos[i] += gVelocity

    @ti.kernel
    def Seek(self):
        """
        agent 的 seek 行为
        agent的目标位置在self.target中
        agent的世界坐标位置在self.pos中
        Seek 设置在 AgentConfig.KSeek 中
        你需要为所有agent计算运动速度和方向，将它们分别存储到self.vd和self.thetad中
        Seek behavior
        Global goal position is in self.target
        Agent's global position is in self.pos
        Seek setting is in AgentConfig.KSeek
        You need to compute the desired velocity and desired orientation
        Store them into vd and thetad respectively
        """
        # 你需要实现此函数的内容
        pass

    @ti.kernel
    def Flee(self):
        """
        agent 的 flee 行为
        agent的目标位置在self.target中
        agent的世界坐标位置在self.pos中
        Flee 设置在 AgentConfig.KFlee 中
        你需要为所有agent计算运动速度和方向，将它们分别存储到self.vd和self.thetad中
        Flee behavior
        Global goal position is in self.target
        Agent's global position is in self.pos
        Flee setting is in AgentConfig.KFlee
        You need to compute the desired velocity and desired orientation
        Store them into vd and thetad respectively
        """
        # 你需要实现此函数的内容
        pass

    @ti.func
    def ArrivalCal(self, i, target):
        """
        agent i的arrival行为，arrival目标为target指定的目标，target是一个ti.vec2
        agent的目标位置在self.target中
        agent的世界坐标位置在self.pos中
        Arrival 设置位于 AgentConfig.KArrival 中
        你需要为agent i计算它的运动速度和方向，将它们分别存储到self.vd[i]和self.thetad[i]中
        还需要使用utils.ToVel()返回一个表示该agent目标速度的向量ti.vec2，其方向为thetad，其范数为vd
        Arrival behavior for agent i with given target:ti.Vec2
        Global goal position is in self.target
        Agent's global position is in self.pos
        Arrival setting is in AgentConfig.KArrival
        You need to compute the desired velocity and desired orientation
        Store them into vd and thetad respectively
        return a ti.vec2 that represents the goal velocity with its direction being thetad and its norm being vd
        """
        # 你需要实现此函数的内容
        pass

    
    @ti.kernel
    def Arrival(self):
        # Arrival behavior
        for i in range(self.n):
            self.ArrivalCal(i,ti.Vector([self.target[0],self.target[1]]))
    
    @ti.kernel
    def Departure(self):
        """
        agent 的departure行为
        agent的目标位置在self.target中
        agent的世界坐标位置在self.pos中
        Departure 设置在 AgentConfig.KDeparture 中
        你需要为所有agent计算运动速度和方向，将它们分别存储到self.vd和self.thetad中
        Departure behavior
        Global goal position is in self.target
        Agent's global position is in self.pos
        Departure setting is in AgentConfig.KDeparture
        You need to compute the desired velocity and desired orientation
        Store them into vd and thetad respectively
        """
        # 你需要实现此函数的内容
        pass

    @ti.func
    def SetV0(self,i):
        """
        在漫游行为计算之前使用
        Use Before Wander Caculation
        input agent id: i
        """
        targetoffset = ti.Vector([self.target[0],self.target[1]])-self.pos[i]
        normv0,_ = utils.Vec2Normalize(targetoffset)
        self.v0[i] = normv0 * AgentConfig.MaxVelocity/1.4
            
    @ti.kernel
    def Wander(self, useNoise: bool):
        """
        agent 的漫游行为
        agent的目标位置在self.target中
        agent的世界坐标位置在self.pos中
        你需要为所有agent计算运动速度和方向，将它们分别存储到self.vd和self.thetad中
        参数useNoise表示此次计算是否要引入噪声。
        如果不引入噪声，则直接使用之前计算得到的vWander，来保持速度的一致性。
        Wander behavior
        Global goal position is in self.target
        Agent's global position is in self.pos
        VWander is in self.vWander
        V0(nominal velocity) is in self.v0, update with self.SetV0(i)
        Wander setting is in AgentConfig.KWander
        You need to compute the desired velocity and desired orientation
        Store them into vd and thetad respectively
        """
        pass

    @ti.kernel
    def Avoid(self):
        """
        agent 的避免行为
        agent的目标位置在self.target中
        agent的世界坐标位置在self.pos中
        障碍物的数量是 self.obstacleNum
        障碍物半径为 AgentConfig.ObstacleRadius
        Avoid设置在 AgentConfig.KAvoid 中
        你需要为所有agent计算运动速度和方向，将它们分别存储到self.vd和self.thetad中
        Avoid behavior
        Global goal position is in self.target
        Agent's global position is in self.pos
        Obstacles are in self.obstacle
        Number of obstacles is self.obstacleNum
        Radius of obstacles is AgentConfig.ObstacleRadius
        Avoid setting is in AgentConfig.KAvoid
        You need to compute the desired velocity and desired orientation
        Store them into vd and thetad respectively
        """
        # 你需要实现此函数的内容
        pass

    @ti.func
    def SeperationCal(self,i):
        """
        agent i的seperation行为
        agent的目标位置在self.target中
        agent的世界坐标位置在self.pos中
        Seperation 设置位于 AgentConfig.KSeperation 和 AgentConfig.RNeighborhood 中
        你需要为agent i计算它的运动速度和方向，将它们分别存储到self.vd[i]和self.thetad[i]中
        还需要使用utils.ToVel()返回一个表示该agent目标速度的向量ti.vec2，其方向为thetad，其范数为vd
        Seperation behavior for agent i
        Global goal position is in self.target
        Agent's global position is in self.pos
        Seperation setting is in AgentConfig.KSeperation and AgentConfig.RNeighborhood
        You need to compute the desired velocity and desired orientation
        Store them into vd and thetad respectively
        return a ti.vec2 that represents the goal velocity with its direction being thetad and its norm being vd
        """
        # 你需要实现此函数的内容
        pass
    
    @ti.kernel
    def Seperation(self):
        # Seperation behavior 
        for i in range(self.n):
            self.SeperationCal(i)

    @ti.func
    def AlignmentCal(self,i):
        """
        agent i的alignment行为
        agent的目标位置在self.target中
        agent的世界坐标位置在self.pos中
        Alignment 设置位于 AgentConfig.KAlignment 中
        你需要为agent i计算它的运动速度和方向，将它们分别存储到self.vd[i]和self.thetad[i]中
        还需要使用utils.ToVel()返回一个表示该agent目标速度的向量ti.vec2，其方向为thetad，其范数为vd
        Alignment behavior for agent i
        Global goal position is in self.target
        Agent's global position is in self.pos
        Alignment setting is in AgentConfig.KAlignment
        You need to compute the desired velocity and desired orientation
        Store them into vd and thetad respectively
        return a ti.vec2 that represents the goal velocity with its direction being thetad and its norm being vd
        """
        # 你需要实现此函数的内容
        pass

    @ti.kernel
    def Alignment(self):
        # Alignment behavior 
        for i in range(self.n):
            self.AlignmentCal(i)

    @ti.func
    def CohensionCal(self,i):
        """
        agent i的cohension行为
        agent的目标位置在self.target中
        agent的世界坐标位置在self.pos中
        Cohension 设置位于 AgentConfig.KCohension 中
        你需要为agent i计算它的运动速度和方向，将它们分别存储到self.vd[i]和self.thetad[i]中
        还需要使用utils.ToVel()返回一个表示该agent目标速度的向量ti.vec2，其方向为thetad，其范数为vd
        Cohension behavior for agent i
        Global goal position is in self.target
        Agent's global position is in self.pos
        Cohension setting is in AgentConfig.KCohension
        You need to compute the desired velocity and desired orientation
        Store them into vd and thetad respectively
        return a ti.vec2 that represents the goal velocity with its direction being thetad and its norm being vd
        """
        # 你需要实现此函数的内容
        pass
    
    @ti.kernel
    def Cohension(self):
        # Cohension behavior
        for i in range(self.n):
            self.CohensionCal(i)

    @ti.kernel
    def Flocking(self):
        """
        试图追逐目标的flocking行为计算
        利用Separation, Cohesion和Alignment行为来确定所需的速度矢量
        调用函数：self.ArrivalCal(i),self.CohensionCal(i),self.SeperationCal(i),self.AlignmentCal(i)
        你需要计算所有agent运动速度和方向，将它们分别存储到 vd 和 thetad 中
        Flocking behavior while arrving target
        Utilize the Separation, Cohesion and Alignment behaviors to determine the desired velocity vector
        Call function: self.ArrivalCal(i),self.CohensionCal(i),self.SeperationCal(i),self.AlignmentCal(i)
        You need to compute the desired velocity and desired orientation
        Store them into vd and thetad respectively
        """
        # 你需要实现此函数的内容
        pass

    @ti.kernel
    def Leader(self):
        """
        领导者到达目标，跟随者跟随领导者
        利用Separation, Cohesion和Alignment行为来确定所需的速度矢量
        调用函数：self.ArrivalCal(i),self.CohensionCal(i),self.SeperationCal(i),self.AlignmentCal(i)
        领导者总是这个类中的第一个agent（id=0），其采用到达行为追逐目标
        跟随者使用leader following行为模式追逐领导者
        追逐目标时候分别需要统计分离和跟随两个力，这两个力的混合倍数分别是AgentConfig.KLeaderSeperation和AgentConfig.KLeaderArrival。
        你需要计算所有agent运动速度和方向，将它们分别存储到 vd 和 thetad 中
        Leader behavior while arrving target
        Utilize the Separation, Cohesion and Alignment behaviors to determine the desired velocity vector
        Call function: self.ArrivalCal(i),self.CohensionCal(i),self.SeperationCal(i),self.AlignmentCal(i)
        You need to find the leader, who is always the first agent in this class(id=0)
        You need to compute the desired velocity and desired orientation
        Store them into vd and thetad respectively
        """
        # 你需要实现此函数的内容
        pass

    def Sense(self,mode):
        if AgentConfig.METHODS[mode] == 'Seek':
            self.Seek()   
        elif AgentConfig.METHODS[mode] == 'Flee':
            self.Flee()
        elif AgentConfig.METHODS[mode] == 'Arrival':
            self.Arrival()
        elif AgentConfig.METHODS[mode] == 'Departure':
            self.Departure()
        elif AgentConfig.METHODS[mode] == 'Wander':
            self.wander_upd = (self.wander_upd + 1) % 15
            if self.wander_upd == 0:
                self.Wander(True)
            else:
                self.Wander(False)
        elif AgentConfig.METHODS[mode] == 'Avoid':
            self.Avoid()
        elif AgentConfig.METHODS[mode] == 'Seperation':
            self.Seperation()
        elif AgentConfig.METHODS[mode] == 'Alignment':
            self.Alignment()
        elif AgentConfig.METHODS[mode] == 'Cohension':
            self.Cohension()
        elif AgentConfig.METHODS[mode] == 'Flocking':
            self.Flocking()
        elif AgentConfig.METHODS[mode] == 'Leader':
            self.Leader()
        
    @ti.kernel
    def UpdateTarget(self,mouse_x:ti.f32,mouse_y:ti.f32):
        # 单击鼠标右键时设置目标
        # set target when right mouse clicked
        assert mouse_x>=0 and mouse_x<=1
        self.target[0] = mouse_x
        assert mouse_y>=0 and mouse_y<=1
        self.target[1] = mouse_y
        for i in range(self.n):
            self.SetV0(i)
        
    def Render(self,gui,mode,WINDOW_HEIGHT):
        gui.clear(0xffffff)
        # target render 渲染目标
        gui.circle([self.target[0],self.target[1]],0xff0000,5)

        centers = self.pos.to_numpy()
        centers = centers[:self.n]
        gui.circles(centers, color=0x000000, radius=10)

        # label render 渲染提示词
        str = 'mode: '+AgentConfig.METHODS[mode]
        help1 = 'PRESS Q:Seek,W:Flee,E:Arrival,R:Departure,T:Wander,Y:Avoid,U:Seperation'
        help2 = '                        I:Alignment,O:Cohension,P:Flocking,SPACE:Leader'
        gui.text(content=str, pos=[0.4,0.9], font_size=20, color=0x0000ff)
        gui.text(content=help1, pos=[0,1], font_size=16, color=0x000000)
        gui.text(content=help2, pos=[0,0.95], font_size=16, color=0x000000)

        # leader render 渲染leader
        if AgentConfig.METHODS[mode] == 'Leader': 
            leadercenter = centers[0]
            gui.circle(leadercenter,0x0000ff,radius=10)

        # render obstacle 渲染障碍物
        if AgentConfig.METHODS[mode] == 'Avoid': 
            obstacle_r = self.obstacle.to_numpy()
            gui.circles(obstacle_r,color=0x00ff00,radius=AgentConfig.ObstacleRadius * WINDOW_HEIGHT)
