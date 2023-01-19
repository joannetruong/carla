import os
import glob
import cv2
import numpy as np
from real_policy import NavPolicy, ContextNavPolicy

policy_type = "NavPolicy"
weights_dir = "/home/joanne/repos/carla/PythonAPI/all/weights/"
if policy_type == "ContextNavPolicy":
    policy_pth = "spot_depth_context_resnet18_map_prevact_sincos32_log_rot_100_-1.0_cma_bp_0.3_sd_1_ckpt_93.pth"
else:
    policy_pth = "spot_depth_simple_cnn_cutout_nhy_2hz_hm3d_mf_rand_pitch_-1.0_1.0_bp_0.03_log_sd_1_ckpt_95_v2.pth"
weights = os.path.join(weights_dir, policy_pth)

# depth_pth = '/home/joanne/repos/carla/PythonAPI/all/eval_imgs/Town01_1'
depth_pth = (
    "/home/joanne/repos/carla/PythonAPI/all/eval/eval_imgs/Town01_1_1674094172.3373778"
)
policy = eval(policy_type)(weights)
policy.reset()

for img in sorted(glob.glob(os.path.join(depth_pth, "depth_*.png"))):
    img_num = img.split("_")[-1][:-4]
    depth_img = cv2.imread(img)[:, :, 0]
    observations = {}
    split_depth_img = np.split(depth_img / 255.0, 2, axis=1)
    observations["spot_left_depth"] = np.expand_dims(split_depth_img[1], 2).astype(
        "float32"
    )
    observations["spot_right_depth"] = np.expand_dims(split_depth_img[0], 2).astype(
        "float32"
    )

    goal_dist = np.exp(4.7)
    goal_heading = -1.416
    observations["pointgoal_with_gps_compass"] = np.array(
        [np.log(goal_dist), goal_heading], dtype=np.float32
    )

    policy_actions = policy.act(observations, deterministic=True)
    base_action = np.clip(policy_actions, -1, 1)
    # Silence low actions
    # base_action[np.abs(base_action) < 0.05] = 0.0

    base_action *= [0.5, 0.52]
    base_vel = base_action * 0.5
    print(
        "img num: ",
        img_num,
        "policy_actions: ",
        base_vel[0],
        np.rad2deg(base_vel[1]),
        observations["pointgoal_with_gps_compass"],
    )
    input()
