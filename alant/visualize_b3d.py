
import os
from typing import List, Dict, Tuple
import nimblephysics as nimble
from nimblephysics import NimbleGUI
import numpy as np
import time
from nimblephysics.server import GUIWebsocketServer


def visualize(dataset_b3d_path):
    geometry = './ml_and_simulation/data/Geometry'
    print('Using Geometry folder: '+geometry)
    geometry = os.path.abspath(geometry)
    if not geometry.endswith('/'):
        geometry += '/'
        print(' > Converted to absolute path: '+geometry)

    subject: nimble.biomechanics.SubjectOnDisk = nimble.biomechanics.SubjectOnDisk(dataset_b3d_path)


    # skel = subject.readSkel(0, geometryFolder=geometry)

    customOsim: nimble.biomechanics.OpenSimFile = nimble.biomechanics.OpenSimParser.parseOsim(
        path + 'unscaled_generic.osim')
    skel = customOsim.skeleton



    world = nimble.simulation.World()
    world.addSkeleton(skel)
    world.setGravity([0, -9.81, 0])
    skel.setGravity([0, -9.81, 0])
    # loaded = subject.readFrames(trial_index, 0, 1)

    gui = NimbleGUI(world)
    gui.serve(8080)
    gui.nativeAPI().renderSkeleton(skel)
    while True:
        for i_trial in range(subject.getNumTrials()):
            num_frames = subject.getTrialLength(i_trial)
            for frame in range(num_frames):
                loaded: List[nimble.biomechanics.Frame] = subject.readFrames(i_trial, frame, 1, contactThreshold=20)
                skel.setPositions(loaded[0].processingPasses[0].pos)
                gui.nativeAPI().renderSkeleton(skel)
                time.sleep(0.001)

sub_name = 'Subject002'
path = f"/mnt/d/Local/AddBiom/vmu-suit/test_data/NordData/{sub_name}/"
# define main function
if __name__ == "__main__":
    dataset_b3d_path = path + sub_name + '.b3d'
    visualize(dataset_b3d_path)



