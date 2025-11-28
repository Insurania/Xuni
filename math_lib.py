"""本文件包含了基本的数据类型和其常用运算

Author:
    lire 

Date:
    2025-10-28

Version:
    1.0
"""

import taichi as ti
import taichi.math as tm
import taichi.types as tt

import math

# for a 3D point and vector
vec3 = ti.types.vector(3, ti.f32)
# for humogeneous coordinate
vec4 = ti.types.vector(4, ti.f32)
# for matrix 3x3, use for scale and rotation
mat3 = ti.types.matrix(3, 3, ti.f32)
# for matrix 4x4, for any affline transformation
mat4 = ti.types.matrix(4, 4, ti.f32)
# quanternion, for rotation
quaternion = ti.types.vector(4, ti.f32)

@ti.func
def mat3_eye() -> mat3:
    return mat3(1, 0, 0, 0, 1, 0, 0, 0, 1)

@ti.func
def axis_x() -> vec3:
    """Get unit vector of axis x

    Returns:
        vec3: [1,0,0]
    """
    return vec3(1.0, 0, 0)

@ti.func
def axis_y() -> vec3:
    """Get unit vector of axis y

    Returns:
        vec3: [0,1,0]
    """
    return vec3(0, 1.0, 0)

@ti.func
def axis_z() -> vec3:
    """Get unit vector of axis z

    Returns:
        vec3: [0,0,1]
    """
    return vec3(0, 0, 1.0)

eps = 1e-4

# quaternion calc
@ti.func
def dot(x: quaternion, y: quaternion) -> ti.f32:
    """Dot product

    Args:
        x (quaternion): x
        y (quaternion): y

    Returns:
        ti.f32: result
    """
    return tm.dot(x, y)

@ti.func
def inv(x: quaternion) -> quaternion:
    """inverse of quaternion

    [w, x, y, z] = [w, -x, -y, -z] / (w^2 + x^2 + y^2 + z^2)

    Args:
        x (quaternion): x

    Returns:
        quaternion: inversed
    """
    return quaternion(x.r, -x.gba) / sqr_length(x)

@ti.func
def mul(x: quaternion, y: quaternion) -> quaternion:
    """quaternion multiply

    q1 = [p q], q2 = [p' q']
    then q1 * q2 = [p * p' - q dot q', pq' + qp' + q cross q']

    Args:
        x (quaternion): x
        y (quaternion): y

    Returns:
        quaternion: result
    """
    return quaternion(x.r * y.r - tm.dot(x.gba, y.gba), x.r * y.gba + y.r * x.gba + tm.cross(x.gba, y.gba))

@ti.func
def div(x: quaternion, y: quaternion) -> quaternion:
    """quaternion divide with a quaternion

    Args:
        x (quaternion): x
        y (float): y

    Returns:
        quaternion: result
    """
    return mul(x, inv(y))

@ti.func
def sqr_length(x: quaternion) -> float:
    """square of length
    
    x = [w,x,y,z], sqr_length = w*w+x*x+y*y+z*z

    Args:
        x (quaternion): x

    Returns:
        float: result
    """
    return dot(x, x)

@ti.func
def length(x: quaternion) -> float:
    """length

    Args:
        x (quaternion): x

    Returns:
        float: result
    """
    return tm.length(x)

@ti.func
def normalize(x: quaternion) -> quaternion:
    """normalized quaternion

    return a unit quaternion

    Args:
        x (quaternion): x

    Returns:
        quaternion: result
    """
    l = length(x)
    q = quaternion(1,0,0,0)
    if l > eps and ti.abs(l) < 1e6:
        q = x / l
    return q

@ti.func
def slerp(x: quaternion, y: quaternion, t: float) -> quaternion:
    """slerp between quaternions

    Algorithm reference eigen: https://github.com/PX4/eigen/blob/master/Eigen/src/Geometry/Quaternion.h

    Args:
        x (quaternion): x
        y (quaternion): y
        t (float): lerp ratio

    Returns:
        quaternion: result
    """
    d = dot(x, y)
    abs_d = ti.abs(d)
    # for sin theta -> 0, use nlerp
    s0 = 1 - t
    s1 = t
    if abs_d < 1.0 - eps:
        theta = tm.acos(abs_d)
        sin_theta = tm.sin(theta)
        s0 = tm.sin((1 - t) * theta) / sin_theta
        s1 = tm.sin(t * theta) / sin_theta
    if d < 0:
        s0 = -s0
    return s0 * x + s1 * y

@ti.func
def exp(x: quaternion) -> quaternion:
    """expotional of quaternion

    exp [p,q] = (cos(theta), sin(theta) / theta * q)
    where theta = |q|

    Args:
        x (quaternion): x (norm vector!!)

    Returns:
        quaternion: result
    """
    angle = tm.length(x.gba)
    s = tm.sin(angle)
    c = tm.cos(angle)
    coeff = s / angle if s > eps else 1
    return quaternion(c, coeff * x.gba)

@ti.func
def log(x: quaternion) -> quaternion:
    """log of quanternion

    log x = log [p,q] = (log p, q/||q|| acos(p/|x|))

    Args:
        x (quaternion): x

    Returns:
        quaternion: result
    """
    sinx = tm.length(x.gba)
    cosx = x.r
    theta = tm.atan2(sinx, cosx)
    return x if cosx > 1 - eps else quaternion(0, x.gba * (theta / sinx))

@ti.func
def calc_squad_controlpoint(q0: quaternion, q1: quaternion, q2: quaternion) -> quaternion:
    """Compute the control point of C1 consistency 3-order Bezier curve for slerp used in SQUAD

    Read https://blog.csdn.net/shao918516/article/details/109738930 for details.

    Args:
        q0 (quaternion): pre point
        q1 (quaternion): current point
        q2 (quaternion): next point

    Returns:
        quaternion: control point
    """
    inved = inv(q1)
    return mul(q1, exp(-0.25 * (log(mul(inved, q0)) + log(mul(inved, q2)))))

@ti.func
def squad(q0: quaternion, a: quaternion, b: quaternion, q1: quaternion, t: float):
    """SQUAD operation

    approximation of three order Bezier curve, and generalize to squad.

    The traditional Bezier curve use de Casteljau algorithm: B(v0, v1, v2, v3, t) = L(L(L(v0,v1,t),L(v1,v2,t),t),L(L(v1,v2,t),L(v2,v3,t),t),t)
    Which need 7 slerp, costy. So shoemake et al.[1987] use such approximation:
    quad(v0, v1, v2, v3, t) = lerp(lerp(v0, v3, t), lerp(v1, v2, t), 2t(1-t))
    Read https://blog.csdn.net/shao918516/article/details/109738930 for details.

    Args:
        q0 (quaternion): left of control point
        a (quaternion): left middle of control point
        b (quaternion): right middle of control point
        q1 (quaternion): right of control point
        t (float): time ratio

    Returns:
        _type_: _description_
    """
    return slerp(slerp(q0, q1, t), slerp(a, b, t), 2 * t * (1 - t))

# conversion between quaternion and different notations

@ti.func
def quaternion_to_axisangle(q: quaternion, angle_rad: ti.template()) -> vec3:
    """Convert quaternion to axis-angle

    The quaternion [cos theta/2, sin theta/2 v] means a rotation around v for theta / 2
    You can read the Rodrigues' Rotation Formula for reason.

    Args:
        q (quaternion): quanternion
        angle_rad (ti.template): angle rad, pass by reference

    Returns:
        vec3: rotation axis
    """
    l = length(q)
    res = vec3(0,0,0)
    angle_rad = 0
    if l > eps and tm.dot(q.yzw, q.yzw) > eps:
        angle_rad = 2 * tm.acos(q.x)
        res = tm.normalize(q.yzw)
    return res

@ti.func
def axisangle_to_quaternion(axis: vec3, angle_rad: float) -> quaternion:
    """Convert axis-angle to quaternion

    The quaternion [cos theta/2, sin theta/2 v] means a rotation around v for theta / 2
    You can read the Rodrigues' Rotation Formula for reason.

    Args:
        axis (vec3): aixs
        angle_rad (float): angle in radius

    Returns:
        quaternion: quaternion
    """
    sn = tm.sin(angle_rad / 2)
    cs = tm.cos(angle_rad / 2)
    return quaternion(cs, sn * axis)


@ti.func
def quaternion_to_matrix(q: quaternion) -> mat3:
    """Convert quaternion to rotation matrix

    Reference *quaternions for computer graphics* for details

    Args:
        q (quaternion): quaternion

    Returns:
        mat3: rotation matrix
    """

    tx  = 2.0 * q.y
    ty  = 2.0 * q.z
    tz  = 2.0 * q.w
    twx = tx * q.x
    twy = ty * q.x
    twz = tz * q.x
    txx = tx * q.y
    txy = ty * q.y
    txz = tz * q.y
    tyy = ty * q.z
    tyz = tz * q.z
    tzz = tz * q.w

    m = mat3([
        [1.0 - tyy - tzz, txy - twz, txz + twy],
        [txy + twz, 1.0 - txx - tzz, tyz - twx],
        [txz - twy, tyz + twx, 1.0 - txx - tyy]
    ])
    return m

@ti.func
def axis_angle_to_mat3(axis: vec3, angle: float) -> mat3:
    """Get Rotation Matrix from axis-angle

    Args:
        axis (vec3): rotation matrix
        angle (float): angle

    Returns:
        mat4: rotation matrix
    """
    c = tm.cos(angle)
    s = tm.sin(angle)
    t = 1.0 - c
    ax_n = tm.normalize(axis)
    return mat3(
        [t * ax_n.x * ax_n.x + c, t * ax_n.x * ax_n.y - s * ax_n.z, t * ax_n.x * ax_n.z + s * ax_n.y],
        [t * ax_n.x * ax_n.y + s * ax_n.z, t * ax_n.y * ax_n.y + c, t * ax_n.y * ax_n.z - s * ax_n.x],
        [t * ax_n.x * ax_n.z - s * ax_n.y, t * ax_n.y * ax_n.z + s * ax_n.x, t * ax_n.z * ax_n.z + c]
    )

@ti.func
def matrix_to_quaternion(rot: mat3) -> quaternion:
    """convert rotation matrix to quaternion

    Algorithm in Ken Shoemake's article in 1987 SIGGraPH course notes
    article "Quaternion Calculus and Fast Animation".

    Args:
        rot (mat3): rotation matrix

    Returns:
        quaternion: result
    """
    ftrace = rot[0,0] + rot[1,1] + rot[2,2]
    res = quaternion([0,0,0,0])
    if ftrace > 0.0:
		# |w| > 1/2, may as well choose w > 1/2
        fRoot = tm.sqrt(ftrace + 1.0)  # 2w
        res.x = 0.5 * fRoot
        fRoot = 0.5 / fRoot  # 1/(4w)
        res.y = (rot[2,1] - rot[1,2]) * fRoot
        res.z = (rot[0,2] - rot[2,0]) * fRoot
        res.w = (rot[1,0] - rot[0,1]) * fRoot
    else:
        # |w| <= 1/2
        i = 0
        j = 1
        k = 2
        if rot[1,1] > rot[0,0]:
            i = 1
            j = 2
            k = 0
        if rot[2,2] > rot[i,i]: 
            i = 2
            j = 0
            k = 1
        fRoot = tm.sqrt(rot[i,i] - rot[j,j] - rot[k,k] + 1.0)
        res[i+1] = .5 * fRoot
        fRoot = .5 / fRoot
        res[0] = (rot[k,j] - rot[j,k]) * fRoot
        res[j+1] = (rot[j,i] + rot[i,j]) * fRoot
        res[k+1] = (rot[k,i] + rot[i,k]) * fRoot
    return normalize(res)

@ti.func
def mat3_to_eular_angle(rot: mat3) -> vec3:
    """convert mat to eular angle

    Args:
        rot (mat3): rotation matrix 3x3

    Returns:
        vec3: eular angles, in radius
    """
    angle_vx = tm.asin(rot[2,1])
    res = vec3([angle_vx,0,0])
    if angle_vx > -tm.pi / 2 + eps:
        if angle_vx < tm.pi / 2 - eps:
            res.z = tm.atan2(-rot[0,1], rot[1,1])
            res.y = tm.atan2(-rot[2,0], rot[2,2])
        else:
            # Gimbol lock
            res.y = 0
            res.z = tm.atan2(rot[0,2], rot[0,0])
    else:
        # Gimbol lock
        res.y = 0
        res.z = -tm.atan2(rot[0,2],rot[0,0])
    return res

@ti.func
def euler_angle_to_mat3(angles: vec3) -> mat3:
    """Euler angle (in radius) zxy to mat3x3

    Args:
        angles (vec3): order in x, y, z, with radius

    Returns:
        mat3: rotation matrix
    """
    return axis_angle_to_mat3(axis_z(), angles.z) @ axis_angle_to_mat3(axis_x(), angles.x) @ axis_angle_to_mat3(axis_y(), angles.y)

# Matrix4x4 operations

@ti.func
def translation_to_mat4(x: vec3) -> mat4:
    """Get Translation Matrix

    Args:
        x (vec3): translation position

    Returns:
        mat4: translation matrix
    """
    return mat4([
        [0, 0, 0, x.r],
        [0, 0, 0, x.g],
        [0, 0, 0, x.b],
        [0, 0, 0, 1]
    ])

@ti.func
def axis_angle_to_mat4(axis: vec3, angle: float) -> mat4:
    """Get Rotation Matrix from axis-angle

    Args:
        axis (vec3): rotation matrix
        angle (float): angle

    Returns:
        mat4: rotation matrix
    """
    c = tm.cos(angle)
    s = tm.sin(angle)
    t = 1.0 - c
    ax_n = tm.normalize(axis)
    return mat4(
        [t * ax_n.x * ax_n.x + c, t * ax_n.x * ax_n.y - s * ax_n.z, t * ax_n.x * ax_n.z + s * ax_n.y, 0],
        [t * ax_n.x * ax_n.y + s * ax_n.z, t * ax_n.y * ax_n.y + c, t * ax_n.y * ax_n.z - s * ax_n.x, 0],
        [t * ax_n.x * ax_n.z - s * ax_n.y, t * ax_n.y * ax_n.z + s * ax_n.x, t * ax_n.z * ax_n.z + c, 0],
        [0, 0, 0, 1]
    )

@ti.func
def scaling_to_mat4(scale: vec3) -> mat4:
    """Get scaling matrix

    Args:
        scale (vec3): scale vector

    Returns:
        mat4: scale matrix
    """
    return mat4(
        [scale.x, 0, 0, 0],
        [0, scale.y, 0, 0],
        [0, 0, scale.z, 0],
        [0, 0, 0, 1]
    )

@ti.func
def construct_transform_matrix(rotation: mat3, translation: vec3) -> mat4:
    """build transform matrix from rotation matrix and translation vector

    Args:
        rotation (mat3): rotation
        translation (vec3): translation

    Returns:
        mat4: Transform Matrix
    """
    return mat4([
        [rotation[0,0], rotation[0,1], rotation[0,2], translation.x],
        [rotation[1,0], rotation[1,1], rotation[1,2], translation.y],
        [rotation[2,0], rotation[2,1], rotation[2,2], translation.z],
        [0, 0, 0, 1]
    ])

@ti.func
def get_rot_matrix(m4: mat4) -> mat3:
    return mat3([[m4[0,0], m4[0,1], m4[0,2]], [m4[1,0], m4[1,1], m4[1,2]], [m4[2,0], m4[2,1], m4[2,2]]])

@ti.func
def get_translation(m4: mat4) -> mat3:
    return vec3(m4[0,3], m4[1,3], m4[2,3])

@ti.kernel
def get_axis_angle_from_matrix4x4(mat: mat4) -> vec4:
    q = matrix_to_quaternion(get_rot_matrix(mat))
    angle = .0
    res = quaternion_to_axisangle(q, angle)
    return vec4(res, angle)

@ti.kernel
def eular_translation_to_transmat(eX: float, eY: float, eZ: float, tX: float, tY: float, tZ: float) -> mat4:
    """Taichi kernel to build transform matrix

    Args:
        eX (float): eular angle x
        eY (float): eular angle y
        eZ (float): eular angle z
        tX (float): translation x
        tY (float): translation y
        tZ (float): translation z

    Returns:
        mat4: transform matrix
    """
    pos = vec3(tX, tY, tZ)
    ang = vec3(eX, eY, eZ) / 180 * tm.pi
    rot = euler_angle_to_mat3(ang)
    return construct_transform_matrix(rot, pos)

@ti.func
def ray_sph_inter(start: vec3, dir: vec3, center: vec3, radius: float) -> bool:
    """Ray-sphere Intersection Function

    Analyze the solution of (s + td - c)^2 - R^2 = 0.

    Args:
        start (vec3): Start of ray
        dir (vec3): Directory of ray
        center (vec3): Center of sphere
        radius (float): Radius of sphere

    Returns:
        bool: True if the ray intersect with sphere
    """
    a = tm.dot(dir, dir)
    b = 2 * tm.dot(dir, start - center)
    c = tm.dot(start - center, start - center) - radius * radius
    return b * b - 4 * a * c >= 0

@ti.kernel
def ray_sphere_list_intersection(start: vec3, dir: vec3, radius: float, list: ti.template()) -> int:
    """Return the closest id of spheres which intersect with ray

    Args:
        start (vec3): start of ray
        dir (vec3): direction of ray
        radius (float): radius of sphere
        list (ti.template): list of vec3, contains spheres

    Returns:
        int: id of spheres
    """
    min_z = 1E5
    res = -1
    for I in range(list.shape[0]):
        pos = list[I]
        if ray_sph_inter(start, dir, pos, radius):
            if pos.z < min_z:
                min_z = pos.z
                res = I
    return res
