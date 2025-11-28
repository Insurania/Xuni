import argparse
import taichi as ti
from enum import Enum

class RunType(Enum):
    CUBE = 1
    FK = 2
    IK = 3
    BLEND = 4
    ROCKET = 5
    CROWD = 6
    
    def __eq__(self, __value: object) -> bool:
        return self.value == __value.value


def main():
    parser = argparse.ArgumentParser(description="Parse argument for animation system")
    parser.add_argument('type', action="store", help="Action type, should be CUBE or FK or IK or BLEND or CROWD or ROCKET", default="FK")
    parser.add_argument('-p', '--step', type=int, metavar='', action='store', dest='step', help='step of cube fill', default=3)
    parser.add_argument('-f', '--file', metavar='', action="store", dest="file", help="Add a BVH file to read", default="./resources/Thief.bvh")
    parser.add_argument('-f1', '--file1', metavar='', action="store", dest="file1", help="Add a BVH file to read", default="./resources/Thief.bvh")
    parser.add_argument('-s', '--frame', type=int, metavar='', action="store", dest="frame", help="The frame number use in IK or BLEND, BVH original data for default", default=-1)
    parser.add_argument('-n', '--crowdnum', type=int, metavar='', dest="crowdnum", help="Set crowd num use in behavior simulation", default=5)
    parser.add_argument('-f2', '--file2', metavar='', action="store", dest="file2", help="Animation 2 for blending", default='./resources/FrightenWalk.bvh')
    parser.add_argument('-t', '--frame2', type=int, metavar='', action="store", dest="frame2", help="The frame number use in BLEND for animation 2", default=1)
    parser.add_argument('-b', '--blendcnt', type=int, metavar='', action='store', dest='blendcnt', help="The frame number for blending", default=10)
    parser.add_argument('-m', '--blendmethod', metavar='', action='store', dest='blendmethod', help="Blend method, should be SLERP, CUBIC or SQUAD", default='SLERP')
    args = parser.parse_args()
    return args 

if __name__ == '__main__':
    args = main()
    ti.init(arch=ti.cpu)
    import ParticleSimulation
    import BehaviorSimulation
    import BoxScene
    from SkeletonDraw import draw_animation
    if args.type == 'FK':
        draw_animation(args.file, RunType.FK)
    elif args.type == 'CUBE':
        BoxScene.box_scene(args.step)
    elif args.type == 'BLEND':
        draw_animation(args.file1, RunType.BLEND, args.frame, args.file2, args.frame2, args.blendcnt, args.blendmethod)
    elif args.type == 'ROCKET':
        ParticleSimulation.ParticleSimulation()
    elif args.type == 'CROWD':
        BehaviorSimulation.BehaviorSimulation(args.crowdnum)
    else:
        draw_animation(args.file, RunType.IK, args.frame)