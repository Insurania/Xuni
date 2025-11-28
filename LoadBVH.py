"""
本文件用来生成BVH数据。
BVH数据是一种通用的动画文件格式，包含了每帧的运动信息。
由于该系统所使用的数据中不存在缩放，只有旋转和位移，
因此忽略了Scale通道。

Author:
    lire 

Date:
    2025-10-23

Version:
    1.0
"""
import io
import os
import re

from NativeSkeleton import ChannelType, NativeJoint, NativeSkeleton
from FK import ForwardKinamics

name_match_reg = r"(?:JOINT|ROOT)\s+(.*?)\s*$"
offset_match_reg = r"OFFSET.*?(-?\d+|-?\d+\.\d+)\s+(-?\d+|-?\d+\.\d+)\s+(-?\d+|-?\d+\.\d+)\s*$"
channel_reg = r"CHANNELS\s+(?:\d)((\s+(\S+))+)"

name_pattern = re.compile(name_match_reg)
offset_pattern = re.compile(offset_match_reg)
channel_pattern = re.compile(channel_reg)

def Recursive_Build_BVH(fs: io.TextIOWrapper, skl: NativeSkeleton, parent: NativeJoint, name: str):
    """Recursive_Build_BVH 递归构建BVH

    Args:
        fs (io.TextIOWrapper): 输入流
        skl (NativeSkeleton): 基本骨骼信息
        parent (NativeJoint): 父关节
        name (str): 当前关节名称
    """
    goff = []
    coff = ""
    has_gen = False
    while True:
        lin = fs.readline().strip()
        if lin == '}':
            break
        if lin == '{':
            continue
        if lin.startswith("OFFSET"):
            off_g = offset_pattern.match(lin)
            goff = [off_g.group(1), off_g.group(2), off_g.group(3)]
        if lin.startswith("CHANNELS"):
            chan_g = channel_pattern.match(lin)
            coff = chan_g.group(1).strip()
        if lin.startswith("JOINT"):
            if not has_gen:
                has_gen = True
                cj = skl.AppendJoint(name, coff, goff, parent)
            name_g = name_pattern.match(lin).group(1)
            Recursive_Build_BVH(fs, skl, cj, name_g)
        if lin.startswith("End Site"):
            if not has_gen:
                has_gen = True
                cj = skl.AppendJoint(name, coff, goff, parent)
            lin = fs.readline()
            lin = fs.readline().strip()
            assert(lin.startswith("OFFSET"))
            off_g = offset_pattern.match(lin)
            goff = [off_g.group(1), off_g.group(2), off_g.group(3)]
            skl.AppendJoint(name + "End", "", goff, cj)
            lin = fs.readline()

def LoadMotionData(fs: io.TextIOWrapper, sk: NativeSkeleton):
    """LoadMotionData 读取运动信息

    Args:
        fs (io.TextIOWrapper): 文件操作符
        sk (NativeSkeleton): 骨骼运动信息
    """
    curLine = ""
    while not curLine.startswith("MOTION"):
        curLine = fs.readline().strip()
    curLine = fs.readline().strip()
    frame = int(re.match(r"Frames:\s+(\d+)$", curLine).group(1))
    curLine = fs.readline().strip()
    interval = float(re.match(r"Frame Time:\s+(\d+\.\d+)$", curLine).group(1))
    sk.frame_count = frame
    sk.interval = interval
    for _ in range(frame):
        curLine = fs.readline().strip().split()
        j = 0
        for s in sk.joints:
            for t in s.channel:
                s.motion[t].append(float(curLine[j].strip()))
                j = j + 1


def ReadBVHFile(filename: str) -> NativeSkeleton:
    """ReadBVHFile 读取BVH文件内容

    Args:
        filename (str): 文件信息

    Returns:
        NativeSkeleton: 骨骼数据
    """
    skeleton = NativeSkeleton()

    with open(filename) as fs:
        curr_line = fs.readline().strip()
        assert(curr_line == "HIERARCHY")
        curr_line = fs.readline().strip()
        assert(curr_line.startswith("ROOT"))
        match = name_pattern.match(curr_line)
        
        root = match.group(1)
        Recursive_Build_BVH(fs, skeleton, None, root)
        skeleton.root = skeleton.joints[0]
        LoadMotionData(fs, skeleton)
        # skeleton.AppendJoint(root, )
    return skeleton