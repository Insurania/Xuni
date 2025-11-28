"""本文件包含了几个BVH数据的基本类型。

export:
    - ChannelType 枚举
    - NativeJoint 关节数据
    - NativeSkeleton 骨骼总数据

Author:
    lire 

Date:
    2025-10-23

Version:
    1.0
"""

from enum import Enum
import numpy as np
import taichi as ti
class ChannelType(Enum):
    Xposition = 1
    Yposition = 2
    Zposition = 3
    Xrotation = 4
    Yrotation = 5
    Zrotation = 6

def StringToChannelType(chanStr: str) -> ChannelType:
    return ChannelType[chanStr]

class NativeJoint(object):
    """一个原生Python类的关节定义

    Args:
        name (str): 关节名

    Attributes:
        channel (list[ChannelType]): 关节通道列表
        parent NativeJoint: 父关节
        child list[NativeJoint]: 子关节列表
        motion dict[ChannelType, list[float]]: 运动信息列表
    """
    def __init__(self, name: str) -> None:
        self.name = name
        self.channel: list[ChannelType] = []
        self.parent: int = -1
        self.child: list[int] = []
        self.offset: list[float] = []
        self.motion: dict[ChannelType, list[float]] = {}

class NativeSkeleton:
    """ 骨骼数据类型

    Attributes:
        joints (list[NativeJoint]): 关节列表
        root (NativeJoint): 根关节
        interval (float): 动画间隔
        frame_count (int): 帧数
    """
    def __init__(self) -> None:
        self.joints :list[NativeJoint] = []
        self.root: NativeJoint = None
        self.frame_count: int = 0
        self.interval: float = .0333
        self.npList = []
        self.hasNp = False

    def AppendJoint(self, name: str, channel: str, offset: list[str], parent: NativeJoint) -> NativeJoint:
        """AppendJoint 添加关节到树中

        Args:
            name (str): 关节名称
            channel (str): 关节通道
            offset (list[str]): 关节坐标偏移
            parent (NativeJoint): 关节父节点

        Returns:
            NativeJoint: 生成关节
        """
        joint = NativeJoint(name)
        if parent != None:
            joint.parent = self.joints.index(parent)
            parent.child.append(len(self.joints))
        joint.offset = list(map(lambda x: float(x), offset))
        joint.channel = [val for val in map(StringToChannelType, channel.strip().split(' '))] if not (channel == "" or channel.isspace()) else []  
        for i in joint.channel:
            joint.motion[i] = []
        self.joints.append(joint)
        return joint

    def RemoveDeepJoint(self):
        toBeRemove = []
        newIdList = []
        ni = 0
        for i in range(len(self.joints)):
            id = 0   
            p = i
            while p != -1:
                # print(i, p, id)
                id = id + 1
                p = self.joints[p].parent
            if id > 7:
                # print(i)
                toBeRemove.append(self.joints[i])
                newIdList.append(-1)
            else:
                newIdList.append(ni)
                ni = ni + 1
        for i in toBeRemove:
            self.joints.remove(i)
        # print(newIdList)
        for i in self.joints:
            i.parent = newIdList[i.parent] if i.parent != -1 else -1 
            for t in range(len(i.child)):
                i.child[t] = newIdList[i.child[t]]
            i.child = [v for v in i.child if v != -1]
            # print(i.name, i.parent, i.child, i.offset)

    def setNpList(self, lst):
        self.hasNp = True
        self.npList.append(lst)