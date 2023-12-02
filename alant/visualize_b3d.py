
import os
from typing import List
import nimblephysics as nimble
from nimblephysics import NimbleGUI
import time


def visualize(dataset_b3d_path):
    geometry = '/mnt/d/Local/AddBiom/vmu-suit/ml_and_simulation/data/Geometry/'
    print('Using Geometry folder: '+geometry)
    geometry = os.path.abspath(geometry)
    if not geometry.endswith('/'):
        geometry += '/'
        print(' > Converted to absolute path: '+geometry)

    subject: nimble.biomechanics.SubjectOnDisk = nimble.biomechanics.SubjectOnDisk(dataset_b3d_path)

    skel = subject.readSkel(0, geometryFolder=geometry)

    world = nimble.simulation.World()
    world.addSkeleton(skel)
    world.setGravity([0, -9.81, 0])
    skel.setGravity([0, -9.81, 0])
    # loaded = subject.readFrames(trial_index, 0, 1)

    gui = NimbleGUI(world)
    gui.serve(8090)
    gui.nativeAPI().renderSkeleton(skel)
    while True:
        for i_trial in range(subject.getNumTrials()):
            num_frames = subject.getTrialLength(i_trial)
            trial_name = subject.getTrialName(i_trial)
            gui.nativeAPI().createText('trial_name', trial_name, [1200, 200], [250, 50])
            for frame in range(num_frames):
                loaded: List[nimble.biomechanics.Frame] = subject.readFrames(i_trial, frame, 1, contactThreshold=20)
                skel.setPositions(loaded[0].processingPasses[0].pos)
                gui.nativeAPI().renderSkeleton(skel)
                time.sleep(0.008)


sub_name = 'Uhlrich2023_Formatted_With_Arm_subject3'
path = '/mnt/d/Local/Data/MotionModelData/train/'
if __name__ == "__main__":
    dataset_b3d_path = path + sub_name + '.b3d'
    visualize(dataset_b3d_path)



