import taichi as ti
from math_lib import *
from enum import Enum
from FK import ForwardKinamics, update_fk

class BlendMethod(Enum):
    """BlendMethod 融合方法枚举

    Returns:
        SLERP: 球面插值方法
        CUBIC: 三次样条插值方法
        SQUAD: 球面三次插值方法
    """
    SLERP = 1
    CUBIC = 2
    SQUAD = 3

    def __eq__(self, __value: object) -> bool:
        return self.value == __value.value

class BlendStat(Enum):
    """BlendStat 当前处于的融合状态枚举

    Returns:
        ANIM1: 播放动画1
        BLENDING: 处于两个动画之间的融合态
        ANIM2: 播放动画2
    """
    ANIM1 = 1
    BLENDING = 2
    ANIM2 = 3

    def __eq__(self, __value: object) -> bool:
        return self.value == __value.value

class AnimationBlender:
    def __init__(self, fk1: ForwardKinamics, fk2: ForwardKinamics, start: int, end: int, duration: int, method: BlendMethod):
        """__init__ 初始化函数

        要求start、end处于中间帧，不能是开始帧和结束帧

        Args:
            fk1 (ForwardKinamics): 动画1
            fk2 (ForwardKinamics): 动画2
            start (int): 动画1开始过渡的帧
            end (int): 动画2结束过渡的帧
            duration (int): 过渡帧数
            method (BlendMethod): 混合方法

        Raises:
            Exception: 动画1的开始过渡帧范围不正确
            Exception: 动画2的结束过渡帧范围不正确
        """
        self.fk1 = fk1
        self.fk2 = fk2
        self.start = start
        self.end = end
        self.duration = duration
        self.method = method
        self.sum_duration = start + duration + fk2.frame_cnt - end
        if start < 1 or start > fk1.frame_cnt - 2:
            raise Exception("Start should be more than 1 and less than frame count - 1")
        if end < 1 or end > fk2.frame_cnt - 2:
            raise Exception("End should be more than 1 and less than frame count - 1")
        self.blend_stat = BlendStat.ANIM1

        """ 几个主要的Field说明：

        假设动画1在帧s开始过渡，过渡到动画2的帧t

        pre_mat: 动画1的帧s-1的所有关节的变换矩阵
        start_mat: 动画1的帧s的所有关节的变换矩阵
        control_pre_qua: 根据动画1的帧s-1、s、s+1生成每个关节的SQUAD用控制四元数

        end_mat: 动画2的帧t的所有关节的变换矩阵
        post_mat: 动画2的帧t+1的所有关节的变换矩阵
        control_post_qua: 根据动画2的帧t-1、t、t+1生成的每个关节的SQUAD用控制四元数

        current_local: 处于融合状态下的当前局部变换矩阵
        current_world: 处于融合状态下的当前世界变换矩阵
        current_pos: 处于融合状态下的当前世界坐标

        root_rot: 一个0D Field，是融合后动画2的根节点旋转矩阵
        root_pos: 一个0D Field，是融合后动画2的根节点变化矩阵
        """

        # 对field进行初始化
        self.pre_mat = None
        if method != BlendMethod.SLERP:
            self.pre_mat = mat4.field(shape=(fk1.joint_cnt,))
        self.control_pre_qua = None
        if method == BlendMethod.SQUAD:
            self.control_pre_qua = quaternion.field(shape=(fk1.joint_cnt,))
        self.start_mat = mat4.field(shape=(fk1.joint_cnt,))
        self.end_mat = mat4.field(shape=(fk1.joint_cnt,))
        self.post_mat = None
        if method != BlendMethod.SLERP:
            self.post_mat = mat4.field(shape=(fk1.joint_cnt,))
        self.control_post_qua = None
        if method == BlendMethod.SQUAD:
            self.control_post_qua = quaternion.field(shape=(fk1.joint_cnt,))
        self.current_local = mat4.field(shape=(fk1.joint_cnt,))
        self.current_world = mat4.field(shape=(fk1.joint_cnt,))
        self.current_pos = vec3.field(shape=(fk1.joint_cnt,))

        # 计算融合后的根节点变换矩阵
        self.root_rot = mat3.field(shape=())
        self.root_pos = vec3.field(shape=()) # a 0D field
        self.pre_mat_field = mat4.field(shape=(2,))
        self.post_mat_field = mat4.field(shape=(2,))
        self.pre_mat_field[0] = fk1.get_transform_mat(start - 1, 0)
        self.pre_mat_field[1] = fk1.get_transform_mat(start, 0)
        self.post_mat_field[0] = fk2.get_transform_mat(end, 0)
        self.post_mat_field[1] = fk2.get_transform_mat(end + 1, 0)
        make_up_post_motion(self.pre_mat_field, self.post_mat_field, duration,
                            self.root_rot, self.root_pos)
        
        # 将fk1、fk2更新到的信息写到对应field中
        if method != BlendMethod.SLERP:
            fk1.update_time_to_field(start - 1)
            copy_field(fk1.local_matrix, self.pre_mat)
        fk1.update_time_to_field(start)
        copy_field(fk1.local_matrix, self.start_mat)
        fk1.update_time_to_field(start + 1)

        fk2.update_time_to_field(end)
        copy_field(fk2.local_matrix, self.end_mat)
        if method != BlendMethod.SLERP:
            fk2.update_time_to_field(end + 1)
            copy_field(fk2.local_matrix, self.post_mat)
        fk2.update_time_to_field(end - 1)

        # 将根节点变换运用到end_mat、post_mat中
        self.root_pos_mat = self.root_pos[None]
        self.root_rot_mat = self.root_rot[None]
        ori_end = self.end_mat[0]
        self.end_mat[0] = modify_root_transform(self.end_mat[0], ori_end, self.root_pos_mat, self.root_rot_mat)
        if method != BlendMethod.SLERP:
            self.post_mat[0] = modify_root_transform(self.post_mat[0], ori_end, self.root_pos_mat, self.root_rot_mat)
        fk2.local_matrix[0] = modify_root_transform(fk2.local_matrix[0], ori_end, self.root_pos_mat, self.root_rot_mat)

        # 对SQUAD的控制四元数进行计算
        if method == BlendMethod.SQUAD:
            calc_squad_control(self.pre_mat, self.start_mat, fk1.local_matrix,
                               fk2.local_matrix, self.end_mat, self.post_mat,
                               self.control_pre_qua, self.control_post_qua)

        # 强制调用来进行编译
        self.update_timestep(0)
        self.update_timestep(start + 1)
        self.update_timestep(end + 1)

    def get_current_matrix(self, joint_id: int) -> mat4:
        """get_current_matrix 获取当前的全局变换矩阵

        Args:
            joint_id (int): 关节编号

        Returns:
            mat4: 对应的变换矩阵
        """
        if self.blend_stat == BlendStat.ANIM1:
            return self.fk1.global_matrix[joint_id]
        elif self.blend_stat == BlendStat.BLENDING:
            return self.current_world[joint_id]
        else:
            return self.fk2.global_matrix[joint_id]
        
    def update_timestep(self, time: int):
        """update_timestep 对时间进行的更新

        对于当前动画的更新，分成三个部分：
        如果当前时间在动画1，那么直接对fk1进行前向动力学计算
        如果当前时间在1和2之间，那么进行过渡计算，并计算current_world
        如果当前时间在动画2，那么将动画2的变换应用到动画2上，然后对fk2进行前向动力学计算

        Args:
            time (int): 当前所在帧
        """
        time = time % self.sum_duration
        # 动画1
        if time <= self.start:
            self.blend_stat = BlendStat.ANIM1
            self.fk1.update_time(time)
        # 融合计算
        elif time < self.start + self.duration:
            self.blend_stat = BlendStat.BLENDING
            lerp_t = float(time - self.start - 1) / self.duration
            if self.method == BlendMethod.SLERP:
                blend_anim_slerp(self.start_mat, self.end_mat, lerp_t, self.current_local)
            elif self.method == BlendMethod.CUBIC:
                blend_anim_cubic(self.pre_mat, self.start_mat, self.end_mat, self.post_mat, lerp_t, self.current_local)
            elif self.method == BlendMethod.SQUAD:
                blend_anim_squad(self.start_mat, self.end_mat, self.control_pre_qua, self.control_post_qua, lerp_t, self.current_local)
            update_fk(self.fk1.parent_id, self.current_local, self.current_world, self.current_pos)
        # 动画2
        else:
            self.blend_stat = BlendStat.ANIM2
            self.fk2.update_time(time - self.start - self.duration + self.end)
            not_applied = self.fk2.get_joint_mat(0)
            original = self.fk2.get_transform_mat(self.end, 0)
            self.fk2.set_joint_mat(0, modify_root_transform(not_applied, original, self.root_pos_mat, self.root_rot_mat))
            update_fk(self.fk2.parent_id, self.fk2.local_matrix, self.fk2.global_matrix, self.fk2.global_position)

@ti.func
def cubic_vec3(d1: vec3, d2: vec3, d3: vec3, d4: vec3, t: float):
    """cubic_vec3 三次插值
    """
    a = d2
    b = d2 - d1
    c = (d3 - d2) * 3 - (d2 - d1) * 2 - (d4 - d3)
    d = (d2 - d3) * 2 + d2 - d1 + d4 - d3
    return a + b * t + c * t * t + d * t * t * t

@ti.kernel
def copy_field(src: ti.template(), dst: ti.template()):
    """copy_field 拷贝taichi field
    """
    for I in ti.grouped(src):
        dst[I] = src[I]

@ti.kernel
def make_up_post_motion(pre_field: ti.template(), post_field: ti.template(),
                        blend_frame_cnt: int, 
                        root_rot: ti.template(), root_pos: ti.template()):
    """make_up_blend_motion 生成后半部分混合动画帧信息

    例如，动画1有20帧，从第11帧开始混合；动画2有40帧，从第6帧开始混合，用10帧过渡。
    那么，最终生成的动画是
    | 动画1 0-10帧 | 过渡 10帧 | 动画2 6-39帧 |

    在上面的例子中，pre_frame是动画1第10帧的所有关节信息，begin_frame是动画1第11帧的所有关节信息
    end_frame是动画2第6帧的所有关节信息，post_frame是动画2第7帧的所有关节信息
    这一函数需要计算的是后半部分的混合动画的旋转矩阵和根节点坐标。

    考虑动画pre_frame->begin_frame的位移是dx，经过时间是dt，速度v=dx/dt
    动画end_frame->post_frame的位移是dx'，经过时间是dt，速度v'=dx'/dt
    那么在混合过程中，移动的距离应该是(|v|+|v'|)/2*l*dt=(|dx|+|dx'|)/2*l
    对旋转，我们不考虑角速度而进行直接拼接。由此可以的出来对于动画2的变换矩阵。

    Args:
        pre_frame (MatrixField): pre_frame[0]是动画1开始混合帧的前一帧的变换矩阵，pre_frame[1]是开始混合帧的变换矩阵
        post_frame (MatrixField): post_frame[0]是动画2开始混合帧的变换矩阵，post_frame[1]是开始混合帧的后一帧的变换矩阵
        blend_frame_cnt (int): 混合帧数
        root_rot (MatrixField): 旋转变换矩阵，通过root_rot[None]写入
        root_pos (VectorField): 旋转变换矩阵，通过root_pos[None]写入
    """

    pre_frame = pre_field[0]
    begin_frame = pre_field[1] 
    end_frame = post_field[0] 
    post_frame = post_field[1]
    
    # 在这里添加你的实现
    # ...
    # 将计算得到的根节点位置和旋转填写到下边的root_pos和root_rot中
    root_pos[None] = vec3(0,0,0)
    root_rot[None] = mat3(1,0,0,0,1,0,0,0,1)

@ti.kernel
def modify_root_transform(curr_transform: mat4, endframe_transform: mat4,
                          root_pos: vec3, root_rotation: mat3) -> mat4:
    """modify_root_transform 计算当前帧根节点变化矩阵的修正后矩阵

    考虑帧t的根节点是T，动画2开始帧的变换是Q
    那么当Q修正后的变化是Q'，T修正后的变化应该也是T'
    将其拆解为坐标和旋转矩阵变化。旋转变化是一致的，直接应用root_rotation；
    而坐标则是对相对于开始帧的差应用变化root_rotation，然后再加上root_pos。

    Args:
        curr_transform (mat4): 当前的变换矩阵
        endframe_transform (mat4): 动画2开始帧的变化矩阵
        root_pos (vec3): 根节点坐标
        root_rotation (mat3): 根节点旋转

    Returns:
        mat4: 当前帧的新根节点变换矩阵
    """
    # 在这里添加你的实现
    # ...
    # 然后将计算得到的新的变换矩阵返回
    return ti.Matrix.identity(dt=ti.f32, n=4)

@ti.kernel
def blend_anim_slerp(start_frame: ti.template(), end_frame: ti.template(), t: float, result: ti.template()):
    """blend_anim_slerp 使用slerp构造当前动画帧

    Args:
        start_frame (MatrixField): 动画1开始帧每个关节的变换矩阵
        end_frame (MatrixField): 动画2结束帧每个关节的变换矩阵
        t (float): 插值的时刻，[0,1]
        result (MatrixField): 插值结果每个关节的变换矩阵
    """
    for s in result:
        # 在这里添加你的实现
        # ...
        # result[s] 表示新的关节s的变换矩阵
        result[s] = ti.Matrix.identity(dt=ti.f32, n=4)

@ti.kernel
def calc_squad_control(pre1: ti.template(), pre2: ti.template(), pre3: ti.template(),
                       post1: ti.template(), post2: ti.template(), post3: ti.template(),
                       ct0: ti.template(), ct1: ti.template()):
    """calc_squad_control 计算SQUAD用的控制点
    """
    for s in pre1:
        p1 = pre1[s]
        p2 = pre2[s]
        p3 = pre3[s]
        l1 = post1[s]
        l2 = post2[s]
        l3 = post3[s]
        s1 = matrix_to_quaternion(get_rot_matrix(p1))
        s2 = matrix_to_quaternion(get_rot_matrix(p2))
        s3 = matrix_to_quaternion(get_rot_matrix(p3))
        t0 = matrix_to_quaternion(get_rot_matrix(l1))
        t1 = matrix_to_quaternion(get_rot_matrix(l2))
        t2 = matrix_to_quaternion(get_rot_matrix(l3))
        ct0[s] = calc_squad_controlpoint(s1, s2, s3)
        ct1[s] = calc_squad_controlpoint(t0, t1, t2)

@ti.kernel
def blend_anim_squad(start_frame: ti.template(), end_frame: ti.template(),
                     ct0: ti.template(), ct1: ti.template(),
                    t: float, result: ti.template()):
    """blend_anim_squad 使用SQUAD进行融合操作

    Args:
        start_frame (ti.template): 开始帧
        end_frame (ti.template): 结束帧
        ct0 (ti.template): 开始控制点
        ct1 (ti.template): 结束控制点
        t (float): 插值时间
        result (ti.template): 计算结果
    """
    for s in start_frame:
        x0 = get_translation(start_frame[s])
        x1 = get_translation(end_frame[s])
        r0 = matrix_to_quaternion(get_rot_matrix(start_frame[s]))
        r1 = matrix_to_quaternion(get_rot_matrix(end_frame[s]))
        si = squad(r0, ct0[s], ct1[s], r1, t)
        xi = tm.mix(x0, x1, t)
        result[s] = construct_transform_matrix(quaternion_to_matrix(si), xi)

@ti.kernel
def blend_anim_cubic(pre_frame: ti.template(), start_frame: ti.template(), 
                     end_frame: ti.template(), post_frame: ti.template(),
                     t: float, result: ti.template()):
    """blend_anim_cubic 使用三次插值获得补帧结果。

    Args:
        pre_frame (ti.template): 开始帧上一帧的结果
        start_frame (ti.template): 开始帧的关节局部变换矩阵
        end_frame (ti.template): 结束帧的关节局部变换矩阵
        post_frame (ti.template): 结束帧后一帧的关节局部变换矩阵
        t (float): 插值时间
        result (ti.template): 计算结果
    """
    for s in start_frame:
        pre1 = pre_frame[s]
        pre2 = start_frame[s]
        last1 = end_frame[s]
        last2 = post_frame[s]
        e0 = mat3_to_eular_angle(get_rot_matrix(pre1))
        e1 = mat3_to_eular_angle(get_rot_matrix(pre2))
        e2 = mat3_to_eular_angle(get_rot_matrix(last1))
        e3 = mat3_to_eular_angle(get_rot_matrix(last2))
        x0 = vec3(pre2[0,3], pre2[1,3], pre2[2,3])
        x1 = vec3(last1[0,3], last1[1,3], last1[2,3])
        ei = cubic_vec3(e0, e1, e2, e3, t)
        ri = euler_angle_to_mat3(ei)
        xi = tm.mix(x0, x1, t)
        result[s] = construct_transform_matrix(ri, xi)
