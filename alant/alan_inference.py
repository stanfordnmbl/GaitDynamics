from args import parse_train_opt
import torch
from alan_model import MotionModel
import nimblephysics as nimble
from nimblephysics import NimbleGUI
import time


def inference(opt):
    render_count = 10
    cond = torch.ones(render_count)
    wavnames = torch.ones(render_count)
    model = MotionModel(opt.feature_type, opt.checkpoint)
    data_tuple = (None, cond, wavnames)

    model.eval()
    _, cond, wavname = data_tuple
    shape = (render_count, model.horizon, model.repr_dim)
    cond = cond.to(model.accelerator.device)
    samples = model.diffusion.generate_samples(
        shape,
        cond[:render_count],
        model.normalizer,
    )

    customOsim: nimble.biomechanics.OpenSimFile = nimble.biomechanics.OpenSimParser.parseOsim(
        '/mnt/d/Local/Data/MotionModelData/model_and_geometry/unscaled_generic_with_arm.osim')
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
        for i_generation, sample in enumerate(samples):
            num_frames = sample.shape[0]
            gui.nativeAPI().createText('generation num', 'Generation: ' + str(i_generation), [1200, 200], [250, 50])
            for frame in range(num_frames):
                skel.setPositions(sample[frame])
                gui.nativeAPI().renderSkeleton(skel)
                time.sleep(0.05)


if __name__ == "__main__":
    opt = parse_train_opt()
    opt.checkpoint = "./train-2.pt"
    inference(opt)















