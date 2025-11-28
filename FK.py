import taichi as ti
import taichi.math as tm
import numpy as np

from math_lib import *
from NativeSkeleton import NativeJoint, NativeSkeleton, ChannelType

class ForwardKinamics:
    """正向动力学解算类

    主要负责从python scope中获取数据并传送到taichi field中进行初始化
    从而得到根据DFS序排列的的local_matrix矩阵，然后再传送到update_fk进行运算。

    """
    def __init__(self, skl: NativeSkeleton, ignore_parent = False):
        self.joint_cnt = len(skl.joints)
        self.parent_id = ti.field(dtype=ti.i32, shape=(self.joint_cnt,))
        self.local_matrix = ti.Matrix.field(4, 4, dtype=ti.f32, shape=(self.joint_cnt,))
        self.global_matrix = ti.Matrix.field(4, 4, dtype=ti.f32, shape=(self.joint_cnt,))
        self.global_position = vec3.field(shape=(self.joint_cnt))
        self.skl = skl
        self.ignore_parent = ignore_parent
        self.frame_cnt = skl.frame_count
        self.global_pos_vec = self.global_matrix.to_numpy()
        if (not self.skl.hasNp):
            for i in range(0, self.frame_cnt):
                self.update_time_to_field_init(i)
                self.skl.setNpList(self.local_matrix.to_numpy())
        for i in range(0, self.joint_cnt):
            if skl.joints[i].parent == None:
                self.parent_id[i] = -1
            else:
                self.parent_id[i] = skl.joints[i].parent
        update_fk(self.parent_id, self.local_matrix, self.global_matrix, self.global_position)


    def get_rot_angle(self, id: int, key: ChannelType, time: int) -> float:
        """get_rot_angle 获得旋转角

        Args:
            id (int): 对应的关节id
            key (ChannelType): 对应的运动通道
            time (int): 时间单元

        Returns:
            float: 返回结果，如果该通道未被k帧则返回0
        """
        if key in self.skl.joints[id].motion.keys():
            return self.skl.joints[id].motion[key][time]
        else:
            return 0
    
    def get_translation(self, id: int, key: ChannelType, time: int, off_id: int) -> float:
        """get_translation 获得旋转角

        Args:
            id (int): 对应的关节id
            key (ChannelType): 对应的运动通道
            time (int): 时间单元
            off_id (int): 偏移中的id

        Returns:
            float: 返回结果，如果该通道未被k帧则返回初始偏移量
        """
        if key in self.skl.joints[id].motion.keys():
            return self.skl.joints[id].motion[key][time]
        else:
            return self.skl.joints[id].offset[off_id]
        
    def update_time_to_field_init(self, frame: int):
        """update_time_to_field 将对应的帧信息填充到field中

        Args:
            frame (int): 帧信息
        """
        for i in range(0, len(self.skl.joints)):
            if self.ignore_parent and i == 0:
                x_pos = y_pos = z_pos = x_rot = y_rot = z_rot = 0
            else:
                x_pos = self.get_translation(i, ChannelType.Xposition, frame, 0)
                y_pos = self.get_translation(i, ChannelType.Yposition, frame, 1)
                z_pos = self.get_translation(i, ChannelType.Zposition, frame, 2)
                x_rot = self.get_rot_angle(i, ChannelType.Xrotation, frame)
                y_rot = self.get_rot_angle(i, ChannelType.Yrotation, frame)
                z_rot = self.get_rot_angle(i, ChannelType.Zrotation,frame)
            mat = eular_translation_to_transmat(x_rot, y_rot, z_rot, x_pos, y_pos, z_pos)
            self.local_matrix[i] = mat
    
    def update_time_to_field(self, frame: int):
        self.local_matrix.from_numpy(self.skl.npList[frame])

    def update_time(self, frame: int):
        """update_time 更新信息

        Args:
            frame (int): 当前帧
        """
        self.update_time_to_field(frame)
        update_fk(self.parent_id, self.local_matrix, self.global_matrix, self.global_position)

    def get_transform_mat(self, frame: int, id: int) -> mat4:
        """get_transform_mat 获取帧frame关节id的变换矩阵

        Args:
            frame (int): 帧id
            id (int): 关节id

        Returns:
            mat4: 变换矩阵
        """
        x_pos = self.get_translation(id, ChannelType.Xposition, frame, 0)
        y_pos = self.get_translation(id, ChannelType.Yposition, frame, 1)
        z_pos = self.get_translation(id, ChannelType.Zposition, frame, 2)
        x_rot = self.get_rot_angle(id, ChannelType.Xrotation, frame)
        y_rot = self.get_rot_angle(id, ChannelType.Yrotation, frame)
        z_rot = self.get_rot_angle(id, ChannelType.Zrotation,frame)
        mat = eular_translation_to_transmat(x_rot, y_rot, z_rot, x_pos, y_pos, z_pos)
        return mat
    
    def update_skl_data(self):
        """update_skl_data 对骨骼信息进行重新解算
        """
        update_fk(self.parent_id, self.local_matrix, self.global_matrix, self.global_position)
        self.global_pos_vec = self.global_matrix.to_numpy()

    def get_joint_mat(self, joint_id: int) -> mat4:
        """get_joint_mat 获取关节变换矩阵

        Args:
            joint_id (int): 关节id

        Returns:
            mat4: 对应变换矩阵
        """
        return self.local_matrix[joint_id]

    def set_joint_mat(self, joint_id: int, transform: vec4):
        """set_joint_mat 设置关节变换矩阵

        Args:
            joint_id (int): 关节id
            transform (vec4): 对应变换矩阵
        """
        self.local_matrix[joint_id] = transform

    def init_state(self, use_frame_id: int):
        """init_state 对骨骼状态初始化

        Args:
            use_frame_id (int): 使用的帧
        """
        if use_frame_id == -1:
            for i in range(0, len(self.skl.joints)):
                x_pos = self.skl.joints[i].offset[0]
                y_pos = self.skl.joints[i].offset[1]
                z_pos = self.skl.joints[i].offset[2]
                x_rot = 0
                y_rot = 0
                z_rot = 0
                mat = eular_translation_to_transmat(x_rot, y_rot, z_rot, x_pos, y_pos, z_pos)
                self.local_matrix[i] = mat
            update_fk(self.parent_id, self.local_matrix, self.global_matrix, self.global_position)
        else:
            self.update_time(use_frame_id % self.skl.frame_count)

@ti.kernel
def update_fk(id_lst: ti.template(), local_mat_field: ti.template(), 
              global_mat_field: ti.template(), global_position: ti.template()):
    """update_fk fk计算函数

    你需要实现此函数的内容，从而完成FK的解算。保证此函数输入的矩阵是按照dfs序排布的，
    也就是说在field中，父关节一定出现在子关节的前面。

    Args:
        id_lst (ScalarField): 一个包含了所有关节的父亲id的Field，如果是-1则表示为根节点。
        local_mat_field (MatrixField): 局部变换矩阵，每个元素是局部变换信息
        global_mat_field (MatrixField): 全局变换矩阵，每个元素是全局变换信息，需要你完成填充
        global_position (VectorField): 坐标矩阵，每个元素是解算后每个关节的世界坐标
    """
    ti.loop_config(serialize=True)
    length = id_lst.shape[0]
    for i in range(length):
        parent_id = id_lst[i]

        if parent_id == -1:
            global_mat_field[i] = local_mat_field[i]
        else:
            global_mat_field[i] = global_mat_field[parent_id] @ local_mat_field[i]

        mat = global_mat_field[i]
        global_position[i] = vec3(mat[0, 3], mat[1, 3], mat[2, 3])