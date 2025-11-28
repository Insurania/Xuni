# 用来做数学库的单元测试。

import taichi as ti
import taichi.math as tm

import math_lib

ti.init(arch=ti.cpu)

@ti.kernel
def execute_unitest_math_lib():
    # x, y stand for two quaternion
    x = math_lib.quaternion([1, 0, 0, 1])
    y = math_lib.quaternion([1, 1, 0, 0])
    print(math_lib.axis_x()) # [1,0,0]
    print(math_lib.dot(x, y)) # 1
    print(math_lib.inv(x)) # [.5, 0, 0, -.5]
    print(math_lib.mul(x, y)) # [1, 1, 1, 1], means the combination of rotations
    print(math_lib.sqr_length(x)) # 2
    print(math_lib.length(x)) # \sqrt{2}
    print(math_lib.normalize(x)) # \frac{1}{\sqrt{2}} [1,0,0,1]
    # after normalize, x stand for rotate z-axis 90 deg, and y rotate x-axis 90 deg
    x = math_lib.normalize(x)
    y = math_lib.normalize(y)
    print(math_lib.div(x, x)) # [1, 0, 0, 0]
    print(math_lib.slerp(x, y, .6)) # [1, .6, 0, .4]
    print(math_lib.exp(x))
    print(math_lib.log(x))
    ang = .0
    print(math_lib.quaternion_to_axisangle(x, ang))
    print(ang)
    axis_conv = math_lib.axisangle_to_quaternion(math_lib.axis_x(), tm.pi / 6)
    print(axis_conv)
    print(math_lib.quaternion_to_matrix(axis_conv))
    mat_conv = math_lib.axis_angle_to_mat3(math_lib.axis_x(), tm.pi / 6)
    print(mat_conv)
    print(math_lib.matrix_to_quaternion(mat_conv))
    print(math_lib.mat3_to_eular_angle(mat_conv))
    print(math_lib.euler_angle_to_mat3(math_lib.vec3([tm.pi / 6, 0, 0])))

@ti.kernel
def exec2():
    q = math_lib.quaternion(0.998947, -0.004020, 0.001119, -0.000516)
    ang = .0
    axis = math_lib.quaternion_to_axisangle(q, ang)
    print(axis, ang)
    rot = math_lib.axis_angle_to_mat3(axis, ang)
    print(rot)
    rot2 = math_lib.quaternion_to_matrix(q)
    print(rot2)

exec2()
# execute_unitest_math_lib()
x = math_lib.vec3(1,1,1)
print(x[0])
