from alant.args import parse_opt, set_with_arm_opt
import torch
from model.alan_model import MotionModel, MotionDataset
import nimblephysics as nimble
from nimblephysics import NimbleGUI
import time
import os
import matplotlib.pyplot as plt


def inpaint(opt):
    model = torch.load(opt.checkpoint)
    repr_dim = model["ema_state_dict"]["input_projection.weight"].shape[1]
    set_with_arm_opt(opt, repr_dim == 56)

    model = MotionModel(opt, repr_dim)
    render_count = 10

    test_dataset = MotionDataset(
        data_path=b3d_path,
        train=False,
        normalizer=model.normalizer,
        max_trial_num=5,
        trial_start_num=0,
        # max_trial_num=1,
        divide_jittery=False,
        opt=opt,
    )
    wins = [test_dataset[i] for i in range(render_count)]

    # masks = torch.zeros_like(wins[0][0])
    # state_true, state_pred_list = model.eval_loop(opt, wins, masks, num_of_generation_per_window=skel_num-1)


    model_name = 'unscaled_generic_no_arm' if not opt.with_arm else 'unscaled_generic_with_arm'
    customOsim: nimble.biomechanics.OpenSimFile = nimble.biomechanics.OpenSimParser.parseOsim(
        f'/mnt/d/Local/Data/MotionPriorData/model_and_geometry/{model_name}.osim')
    colors = [(0.4, 0.4, 0.4, 1)] + [(0.2, 0.5, 0.5, 1) for _ in range(1, skel_num)]
    skels = [customOsim.skeleton for _ in range(skel_num)]

    skel_pelvis_offsets = [(0, 0, 0.5 * i) for i in range(1, skel_num)]
    for i_skel in range(skel_num-1):
        for i_dof in range(len(skel_pelvis_offsets[i_skel])):
            state_pred_list[i_skel][:, :, i_dof+3] += skel_pelvis_offsets[i_skel][i_dof]

    pose_col_loc = [i_dof for i_dof, dof in enumerate(opt.osim_dof_columns) if '_force_' not in dof]
    force_col_loc = [i_dof for i_dof, dof in enumerate(opt.osim_dof_columns) if '_force_' in dof]

    world = nimble.simulation.World()
    world.setGravity([0, -9.81, 0])
    gui = NimbleGUI(world)
    gui.serve(8090)
    while True:
        gui.nativeAPI().createText('generation num', 'Generation: 0', [1200, 200], [250, 50])
        for i_generation in range(render_count):
            gui.nativeAPI().setTextContents('generation num', 'Generation: ' + str(i_generation))
            num_frames = state_true[0].shape[0]
            for i_frame in range(num_frames):
                for i_skel, states in enumerate([state_true] + state_pred_list):
                    poses = states[i_generation, i_frame, pose_col_loc]
                    skels[i_skel].setPositions(poses)
                    gui.nativeAPI().renderSkeleton(skels[i_skel], prefix='skel' + str(i_skel), overrideColor=colors[i_skel])
                    for i_f, contact_body in enumerate(['calcn_r', 'calcn_l']):
                        body_pos = skels[i_skel].getBodyNode(contact_body).getWorldTransform().translation()
                        forces = states[i_generation, i_frame, force_col_loc[3 * i_f:3 * (i_f + 1)]]
                        gui.nativeAPI().createLine(f'line_{i_skel}_{i_f}', [body_pos, body_pos + 0.1 * forces.numpy()], color=[1, 0., 0., 1])
                time.sleep(0.05)


li, carter, falisse, moore, tan2021, tan2022 = 'li', 'carter', 'falisse', 'moore', 'tan2021', 'tan2022'
uhlrich, santos, vanderzee, wang = 'uhlrich', 'santos', 'vanderzee', 'wang'
b3d_path = f'/mnt/d/Local/Data/MotionPriorData/{li}_dset/'

if __name__ == "__main__":
    skel_num = 2
    opt = parse_opt()
    opt.checkpoint = os.path.dirname(os.path.realpath(__file__)) + "/trained_models/train-3000.pt"
    inpaint(opt)















