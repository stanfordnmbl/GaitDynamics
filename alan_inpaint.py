from alant.args import parse_train_opt, set_with_arm_opt
import torch
from model.alan_model import MotionModel, MotionDataset, inverse_convert_addb_state_to_model_input
import nimblephysics as nimble
from nimblephysics import NimbleGUI
import time
import os


def inference(opt):
    render_count = 10
    cond = torch.ones(render_count)
    wavnames = torch.ones(render_count)

    model = torch.load(opt.checkpoint)
    repr_dim = model["ema_state_dict"]["input_projection.weight"].shape[1]
    set_with_arm_opt(opt, repr_dim == 50)

    model = MotionModel(opt, repr_dim)
    data_tuple = (None, cond, wavnames)

    model.eval()
    _, cond, wavname = data_tuple
    shape = (render_count, model.horizon, model.repr_dim)
    print(shape)
    cond = cond.to(model.accelerator.device)

    test_dataset = MotionDataset(
        data_path=b3d_path,
        train=False,
        joints_3d=opt.joints_3d,
        osim_dof_columns=opt.osim_dof_columns,
        target_sampling_rate=opt.target_sampling_rate,
        normalizer=model.normalizer,
        trial_start_num=10,
        max_trial_num=render_count
    )

    pose_true = [test_dataset[i][0] for i in range(render_count)]
    pose_true = torch.stack(pose_true)
    masks = torch.zeros_like(pose_true)      # 0 for masking, 1 for unmasking
    masks[:, :, 3:9] = 1        # unmask ankle and knee joints
    masks[:, :, -12:] = 1        # unmask hip joints
    constraint = {'mask': masks, 'value': pose_true}

    model_name = 'unscaled_generic_no_arm' if not opt.with_arm else 'unscaled_generic_with_arm'
    customOsim: nimble.biomechanics.OpenSimFile = nimble.biomechanics.OpenSimParser.parseOsim(
        f'/mnt/d/Local/Data/MotionPriorData/model_and_geometry/{model_name}.osim')

    pose_true = model.normalizer.unnormalize(pose_true)
    pose_true = inverse_convert_addb_state_to_model_input(pose_true, opt.model_states_column_names,
                                                          opt.joints_3d, opt.osim_dof_columns)

    colors = [(0.4, 0.4, 0.4, 1)] + [(0.2, 0.5, 0.5, 1) for _ in range(1, skel_num)]
    skels = [customOsim.skeleton for _ in range(skel_num)]

    pose_pred = [model.diffusion.generate_samples(
        shape,
        cond[:render_count],
        model.normalizer,
        opt,
        mode="inpaint",
        constraint=constraint)
        for i_skel in range(1, skel_num)]

    skel_pelvis_offsets = [(0, 0, 0.5 * i) for i in range(1, skel_num)]
    for i_skel in range(skel_num-1):
        for i_dof in range(len(skel_pelvis_offsets[i_skel])):
            pose_pred[i_skel][:, :, i_dof+3] += skel_pelvis_offsets[i_skel][i_dof]

    world = nimble.simulation.World()
    world.setGravity([0, -9.81, 0])
    gui = NimbleGUI(world)
    gui.serve(8090)
    while True:
        gui.nativeAPI().createText('generation num', 'Generation: 0', [1200, 200], [250, 50])
        for i_generation in range(render_count):
            gui.nativeAPI().setTextContents('generation num', 'Generation: ' + str(i_generation))
            num_frames = pose_true[0].shape[0]
            for i_frame in range(num_frames):
                for i_skel, poses in enumerate([pose_true] + pose_pred):

                    # # align pelvis orientation for better visualization
                    # poses[i_generation, i_frame][2] = pose_true[i_generation, i_frame, 2]

                    skels[i_skel].setPositions(poses[i_generation, i_frame])
                    gui.nativeAPI().renderSkeleton(skels[i_skel], prefix='skel' + str(i_skel), overrideColor=colors[i_skel])

                time.sleep(0.015)


li_dataset_path = '/mnt/d/Local/Data/MotionPriorData/train/li_dset/'
# hammer_dset = '/mnt/d/Local/Data/MotionPriorData/train/hammer_dset/'
falisse_dset = '/mnt/d/Local/Data/MotionPriorData/train/falisse_dset/'
tan_dataset_path = '/mnt/d/Local/Data/MotionPriorData/train/tan_dset/'
uhlrich_dataset_path = '/mnt/d/Local/Data/MotionPriorData/train/uhlrich_dset/'

if __name__ == "__main__":
    skel_num = 2
    b3d_path = tan_dataset_path
    opt = parse_train_opt()
    opt.checkpoint = os.path.dirname(os.path.realpath(__file__)) + "/trained_models/train-80.pt"
    inference(opt)















