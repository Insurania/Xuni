import taichi as ti
import numpy as np
# ti.init(arch=ti.cpu,debug=True)

M_PI = 3.14159265358979323846
GRAVITY = 9.8
RAD = M_PI/180.0
# vec3 is a built-in vector type suppied in the `taichi.math` module
vec2 = ti.math.vec2

def randomvector(n):
    components = [np.random.normal() for _ in range(n)]
    r = np.sqrt(sum(x * x for x in components))
    v = np.array([x / r for x in components])
    return v

@ti.func
def Truncate(value, minV, maxV):
    result = value
    if value < minV:
        result = minV
    elif value > maxV:
        result = maxV
    return result

@ti.func
def ToVel(speed,angle):
    return ti.Vector([speed*ti.cos(angle),speed*ti.sin(angle)])

def ToVelPython(speed,angle):
    return vec2(speed*ti.cos(angle),speed*ti.sin(angle))

@ti.func
def SqrtLength(vec):
    return ti.sqrt(vec[0]*vec[0]+vec[1]*vec[1])

@ti.func
def Vec2Normalize(vec):
    norm = vec.norm()
    new_vec = ti.Vector([0.0, 0.0])
    if norm != 0:
        new_vec = vec / norm
    return new_vec, norm

_max = int(0xBBBBBB)
_min = int(0x111111)
_range = _max - _min + 1
def ConvertData(_data):
    """将_data[0,1]映射到不同的hex色彩区间中"""
    hex_color = int(_data * _range + _min)
    return hex_color
