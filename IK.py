import taichi as ti

from math_lib import *
from FK import ForwardKinamics
from NativeSkeleton import NativeJoint

import numpy as np

max_iter_count = 5
max_tolerance = .1

class InverseKinamics:
    def __init__(self, fk: ForwardKinamics):
        self.fk: ForwardKinamics = fk
        self.sel_joint: int = -1
        self.sel_point: vec3 = vec3(0,0,0)
        self.chain_count: int = 0
        self.mask = ti.field(shape=(1,), dtype=int)
        self.chain_matrix_global: ti.MatrixField = mat4.field(shape=(1,))
        self.chain_matrix_local: ti.MatrixField = mat4.field(shape=(1,))
        self.sel_local: mat4 = mat4(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
        self.temp_joint_lst: [int] = []
        self.can_ik_solve = False

    def set_select_joint_id(self, id: int):
        """set_select_joint_id 设置选择的关节，并生成对应的解算链

        Args:
            id (int): 当前选择的关节id
        """
        self.sel_joint = id
        self.temp_joint_lst = []
        if id == -1:
            return
        curr_joint: int = id
        # use this to stop before root, guarentee the root is not move
        # curr_joint = self.fk.skl.joints[curr_joint.parent]
        self.sel_local = self.fk.local_matrix[curr_joint]
        curr_joint = self.fk.skl.joints[curr_joint].parent
        if curr_joint == -1 or curr_joint == 0:
            self.can_ik_solve = False
            return
        self.can_ik_solve = True
        while curr_joint != -1 and self.fk.skl.joints[curr_joint].parent != -1:
            self.temp_joint_lst.append(curr_joint)
            curr_joint = self.fk.skl.joints[curr_joint].parent
        self.chain_count = len(self.temp_joint_lst)
        self.chain_matrix_global = mat4.field(shape=(self.chain_count,))
        self.chain_matrix_local = mat4.field(shape=(self.chain_count,))
        mask_tmp = np.array(self.temp_joint_lst)
        gm = self.fk.global_matrix.to_numpy()
        lm = self.fk.local_matrix.to_numpy()
        temp_np_array_g = np.zeros((self.chain_count,4,4), dtype=np.float32)
        temp_np_array_l = np.zeros((self.chain_count,4,4), dtype=np.float32)
        for i in range(self.chain_count):
            temp_np_array_g[i] = gm[self.temp_joint_lst[i]]
            temp_np_array_l[i] = lm[self.temp_joint_lst[i]]
        self.sel_point = vec3(temp_np_array_g[0][0][3], temp_np_array_g[0][1][3], temp_np_array_g[0][2][3])
        self.chain_matrix_global.from_numpy(temp_np_array_g)
        self.chain_matrix_local.from_numpy(temp_np_array_l)
        self.mask = ti.field(dtype=int, shape=(self.chain_count,))
        self.mask.from_numpy(mask_tmp)

    def is_in_ik_chain(self, joint_id: int) -> bool:
        """is_in_ik_chain 判断关节id是否在chain

        Args:
            joint_id (int): 关节id

        Returns:
            bool: 是否在chain
        """
        return joint_id in self.temp_joint_lst

    def set_target_position(self, pos: vec3):
        """set_target_position 设置关节上位置的信息

        Args:
            pos (vec3): 目标位置
        """
        # print("Target: ", pos)
        if dist_kernel(pos, self.sel_point) < .001 or not self.can_ik_solve:
            return
        solve_ik(pos, self.sel_local, self.fk.global_matrix[0], self.chain_matrix_local, self.chain_matrix_global)
        copy_matrix_from_mask(self.fk.global_matrix, self.chain_matrix_global, self.mask)
        copy_matrix_from_mask(self.fk.local_matrix, self.chain_matrix_local, self.mask)
        self.fk.update_skl_data()
        copy_back_matrix_from_mask(self.fk.global_matrix, self.chain_matrix_global, self.mask)
        copy_back_matrix_from_mask(self.fk.local_matrix, self.chain_matrix_local, self.mask)

@ti.kernel
def dist_kernel(x: vec3, y: vec3) -> float:
    """dist_kernel 求两个vec3之间的距离的kernel

    Args:
        x (vec3)
        y (vec3)

    Returns:
        float: 距离
    """
    return tm.distance(x, y)

@ti.kernel
def solve_ik(target: vec3, end_local: mat4, root_global: mat4, 
             ik_chain_local: ti.template(), ik_chain_global: ti.template()):
    """solve_ik 解算IK

    你需要完成下面的IK解算过程。假设当前的关节链是 0 -> 1 -> 2 -> 3 -> 4，
    选中了3号关节，那么在IK链中存储的实际上只有1、2号矩阵，并按照从根到端的顺序排列。
    而root_global存储了0号关节的全局变换矩阵，end_local存储了4号关节的局部变换矩阵。

    Args:
        target (vec3): IK匹配的目标位置
        end_local (mat4): IK链最末关节相对上一个关节的局部变换矩阵
        root_global (mat4): IK链第一个关节的世界坐标矩阵
        ik_chain_local (ti.template): IK链局部变换矩阵 
        ik_chain_global (ti.template): IK链全局变换矩阵，未储存末端节点信息 hty2025
    """
    iter = 0
    last_end_mat = ik_chain_global[0] 
    while iter < max_iter_count and tm.length(
          get_translation(last_end_mat@ end_local) - target) > max_tolerance:   #hty2025
        cid = 0
        while cid < ik_chain_local.shape[0]:
            # 在这里写你的实现，最终应该写入到ik_chain_local[cid]上
            # .....
            
            ik_chain_local[cid] = mat4([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
            for t in range(ik_chain_global.shape[0]):
                i = ik_chain_global.shape[0] - 1 - t
                if i == ik_chain_global.shape[0] - 1:
                    ik_chain_global[i] = root_global @ ik_chain_local[i]
                else:
                    ik_chain_global[i] = ik_chain_global[i + 1] @ ik_chain_local[i]
                if t == ik_chain_global.shape[0] - 1:
                    last_end_mat = ik_chain_global[i]
            cid = cid + 1
        iter = iter + 1

@ti.kernel
def copy_matrix_from_mask(original: ti.template(), new_create: ti.template(), mask: ti.template()):
    """copy_matrix_from_mask 将矩阵从ik链中提取回去

    Args:
        original (ti.template): 原始矩阵
        new_create (ti.template): 新矩阵
        mask (ti.template): 新矩阵对应原始矩阵的编号
    """
    for i in range(new_create.shape[0]):
        original[mask[i]] = new_create[i]

@ti.kernel
def copy_back_matrix_from_mask(original: ti.template(), new_create: ti.template(), mask: ti.template()):
    """copy_matrix_from_mask 将矩阵fk解算结果提取回ik链

    Args:
        original (ti.template): 原始矩阵
        new_create (ti.template): 新矩阵
        mask (ti.template): 新矩阵对应原始矩阵的编号
    """
    for i in range(new_create.shape[0]):
        new_create[i] = original[mask[i]]