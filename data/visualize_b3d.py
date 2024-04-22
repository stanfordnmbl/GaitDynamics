import os
from typing import List
import nimblephysics as nimble
import numpy as np
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

    gui = NimbleGUI(world)
    gui.serve(8090)
    gui.nativeAPI().renderSkeleton(skel)
    gui.nativeAPI().createText('trial_name', '', [1200, 200], [250, 50])

    while True:
        print('Num trials: ' + str(subject.getNumTrials()))
        for i_trial in range(0, subject.getNumTrials()):
            print('Trial: ' + str(i_trial))
            num_frames = subject.getTrialLength(i_trial)
            trial_name = subject.getTrialName(i_trial)
            gui.nativeAPI().setTextContents('trial_name', trial_name)
            for frame in range(num_frames):
                loaded: List[nimble.biomechanics.Frame] = subject.readFrames(i_trial, frame, 1, contactThreshold=20)
                skel.setPositions(loaded[0].processingPasses[0].pos)
                gui.nativeAPI().renderSkeleton(skel)
                time.sleep(0.008)


sub_name = 'subject2'
path = '/mnt/d/Local/Data/MotionPriorData/b3d_no_arm/cleaned/'
if __name__ == "__main__":
    dataset_b3d_path = path + sub_name + '.b3d'
    visualize(dataset_b3d_path)



