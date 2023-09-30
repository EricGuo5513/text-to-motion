# Installation Notes on Mac M2

Notes:

- **Mac M2 (ARM64) doesn't have conda support for python-3.7** (earliest supported is python-3.8)
- M2 does not work CUDA (which requires NVIDIA hardware): therefore no `tensorrt`

## Install conda and python-3.11 dependencies

- create the conda env with python-3.11
```
conda create --name text2motion python=3.11
```
- install python packages (note that I had to comment out `tensorrt` from the [requirements.txt](./requirements.txt) file)
```
pip install -r requirements.txt
```

# Amendments to Linux Installation

## install conda env

```
conda env  create -f environment.yaml
```

## Unzip into Checkpoints

```
╭─   ~/src/aitok/text-to-motion/checkpoints   main !1 ·································································  text2motion_pub seki@Legion-Ubuntu  21:56:54
╰─❯ ls -lh
total 1.4G
-rw-rw-r-- 1 seki seki 672M Sep 28 22:00 kit.zip
-rw-rw-r-- 1 seki seki 674M Sep 28 22:00 t2m.zip

╭─   ~/src/aitok/text-to-motion/checkpoints   main !1 ·································································  text2motion_pub seki@Legion-Ubuntu  22:02:58
╰─❯ unzip t2m.zip 
Archive:  t2m.zip
   creating: t2m/
...

╭─   ~/src/aitok/text-to-motion/checkpoints   main !1 ·································································  text2motion_pub seki@Legion-Ubuntu  22:03:50
╰─❯ unzip kit.zip 
Archive:  kit.zip
   creating: kit/
```

# Test Run

## First Test

```
╭─   ~/src/aitok/text-to-motion   main !1 ?1 ···································································· ✘ INT  text2motion_pub seki@Legion-Ubuntu  17:05:05
╰─❯ python gen_motion_script.py --name Comp_v6_KLD01 --text_file input.txt --repeat_time 3 --ext customized --gpu_id 0
------------ Options -------------
batch_size: 1
checkpoints_dir: ./checkpoints
dataset_name: t2m
decomp_name: Decomp_SP001_SM001_H512
dim_att_vec: 512
dim_dec_hidden: 1024
dim_movement_dec_hidden: 512
dim_movement_enc_hidden: 512
dim_movement_latent: 512
dim_pos_hidden: 1024
dim_pri_hidden: 1024
dim_text_hidden: 512
dim_z: 128
est_length: False
estimator_mod: bigru
ext: customized
gpu_id: 0
is_train: False
max_text_len: 20
n_layers_dec: 1
n_layers_pos: 1
n_layers_pri: 1
name: Comp_v6_KLD01
num_results: 40
repeat_times: 3
result_path: ./eval_results/
split_file: test.txt
start_mov_len: 10
text_enc_mod: bigru
text_file: input.txt
unit_length: 4
which_epoch: latest
-------------- End ----------------
Total number of descriptions 5
Loading model: Epoch 344 Schedule_len 049
Generate Results
00_005
('A person jumps up and down.',)
01_005
('Someone get on all fours and crawls around.',)
02_005
('A man walks forward, sits on the ground and crosses his legs.',)
03_005
('A person is crawling backwards slowly.',)
04_005
('A person is practicing swimming.',)
Animation Results
00_005
01_005
02_005
03_005
04_005

╭─   ~/src/aitok/text-to-motion   main !1 ?1 ·································································  1m 24s  text2motion_pub seki@Legion-Ubuntu  17:06:49
╰─❯ 
```

The results are in the `eval_results` folder:
```
╭─   ~/src/aitok/text-to-motion/eval_results/t2m/Comp_v6_KLD01   main !1 ?1 ···········································  text2motion_pub seki@Legion-Ubuntu  17:08:41
╰─❯ tree
.
├── customized
│   ├── animations
│   │   ├── C000
│   │   │   ├── gen_motion_00_L068_00_a.mp4
│   │   │   ├── gen_motion_00_L068_00_a.npy
│   │   │   ├── gen_motion_01_L072_00_a.mp4
│   │   │   ├── gen_motion_01_L072_00_a.npy
│   │   │   ├── gen_motion_02_L080_00_a.mp4
│   │   │   └── gen_motion_02_L080_00_a.npy
│   │   ├── C001
│   │   │   ├── gen_motion_00_L164_00_a.mp4
│   │   │   ├── gen_motion_00_L164_00_a.npy
│   │   │   ├── gen_motion_01_L196_00_a.mp4
│   │   │   ├── gen_motion_01_L196_00_a.npy
│   │   │   ├── gen_motion_02_L156_00_a.mp4
│   │   │   └── gen_motion_02_L156_00_a.npy
│   │   ├── C002
│   │   │   ├── gen_motion_00_L196_00_a.mp4
│   │   │   ├── gen_motion_00_L196_00_a.npy
│   │   │   ├── gen_motion_01_L192_00_a.mp4
│   │   │   ├── gen_motion_01_L192_00_a.npy
│   │   │   ├── gen_motion_02_L192_00_a.mp4
│   │   │   └── gen_motion_02_L192_00_a.npy
│   │   ├── C003
│   │   │   ├── gen_motion_00_L196_00_a.mp4
│   │   │   ├── gen_motion_00_L196_00_a.npy
│   │   │   ├── gen_motion_01_L196_00_a.mp4
│   │   │   ├── gen_motion_01_L196_00_a.npy
│   │   │   ├── gen_motion_02_L192_00_a.mp4
│   │   │   └── gen_motion_02_L192_00_a.npy
│   │   └── C004
│   │       ├── gen_motion_00_L196_00_a.mp4
│   │       ├── gen_motion_00_L196_00_a.npy
│   │       ├── gen_motion_01_L196_00_a.mp4
│   │       ├── gen_motion_01_L196_00_a.npy
│   │       ├── gen_motion_02_L196_00_a.mp4
│   │       └── gen_motion_02_L196_00_a.npy
│   └── joints
│       ├── C000
│       │   ├── gen_motion_00_L068.npy
│       │   ├── gen_motion_01_L072.npy
│       │   └── gen_motion_02_L080.npy
│       ├── C001
│       │   ├── gen_motion_00_L164.npy
│       │   ├── gen_motion_01_L196.npy
│       │   └── gen_motion_02_L156.npy
│       ├── C002
│       │   ├── gen_motion_00_L196.npy
│       │   ├── gen_motion_01_L192.npy
│       │   └── gen_motion_02_L192.npy
│       ├── C003
│       │   ├── gen_motion_00_L196.npy
│       │   ├── gen_motion_01_L196.npy
│       │   └── gen_motion_02_L192.npy
│       └── C004
│           ├── gen_motion_00_L196.npy
│           ├── gen_motion_01_L196.npy
│           └── gen_motion_02_L196.npy
└── default
    ├── animations
    └── joints

16 directories, 45 files

```
