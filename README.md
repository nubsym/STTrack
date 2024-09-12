# STTRACK: Joint Spatio-Temporal Modeling for Visual Tracking
The official implementation of STTRACK(https://www.sciencedirect.com/science/article/pii/S0950705123009565).

![STARK_Framework](tracking/framework.pdf)

## Brief Introduction

Similarity-based approaches have made significant progress in visual object tracking (VOT). Although these methods work well in simple scenes, they ignore the continuous spatio-temporal connection of the target in the video sequence. For this reason, tracking by spatial matching solely can lead to tracking failures because of distractors and occlusion. In this paper, we propose a spatio-temporal joint-modeling tracker named STTrack which implicitly builds continuous connections between the temporal and spatial aspects of the sequence. Specifically, we first design a time-sequence iteration strategy (TSIS) to concentrate on the temporal connection of the target in the video sequence. Then, we propose a novel spatial temporal interaction Transformer network (STIN) to capture the spatial-temporal correlation of the target between frames. The proposed STIN module is robust in target occlusion because it explores the dynamic state change dependencies of the target. Finally, we introduce a \textit{spatio-temporal query} to suppress distractors by iteratively propagating the target prior. Extensive experiments on six tracking benchmark datasets demonstrate that the proposed STTrack achieves excellent performance while operating in real-time.

| Tracker | LaSOT (AUC)| GOT-10K (AO)| TrackingNet (AUC)| UAV123 (AUC)| TNL2k (AUC)| LaSO_ext (AUC)|
|---|---|---|---|---|---|---|
|STTrack|67.4|70.8|82.3|69.4|54.4|47.8|

## Preparation
1. Environment
```
conda create -n sttrack python=3.9
conda activate sttrack
bash install_pytorch17.sh
```
2. Data
   (Put the tracking datasets in ./data. It should look like:)
   ```
   ${STTrack_ROOT}
    -- data
        -- lasot
            |-- airplane
            |-- basketball
            |-- bear
            ...
        -- got10k
            |-- test
            |-- train
            |-- val
        -- coco
            |-- annotations
            |-- images
        -- trackingnet
            |-- TRAIN_0
            |-- TRAIN_1
            ...
            |-- TRAIN_11
            |-- TEST
   ```
3. Set project paths
   (Run the following command to set paths for this project)
```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir .
```
  After running this command, you can also modify paths by editing these two files
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```

## Train STTrack
Training with multiple GPUs using DDP
```
# STTrack
python tracking/train.py --script sttrack --config baseline --save_dir . --mode multiple --nproc_per_node 8
```
(Optionally) Debugging training with a single GPU
```
python tracking/train.py --script sttrack --config baseline --save_dir . --mode single
```
## Test and evaluate STTrack on benchmarks

- LaSOT
```
python tracking/test.py sttrack baseline --dataset lasot --threads 32
python tracking/analysis_results.py # need to modify tracker configs and names
```
- GOT10K-test
```
python tracking/test.py sttrack baseline_got10k_only --dataset got10k_test --threads 32
python lib/test/utils/transform_got10k.py --tracker_name sttrack --cfg_name baseline_got10k_only
```
- TrackingNet
```
python tracking/test.py sttrack baseline --dataset trackingnet --threads 32
python lib/test/utils/transform_trackingnet.py --tracker_name sttrack --cfg_name baseline
```
- UAV123, TNL2k, LaSOT_ext
```
python tracking/test.py sttrack baseline --dataset uav --threads 32
python tracking/test.py sttrack baseline --dataset tnl2k --threads 32
python tracking/test.py sttrack baseline --dataset lasot_extension_subset --threads 32
python tracking/analysis_results.py # need to modify tracker configs and names
```

## Acknowledgments
* Thanks for [PyTracking](https://github.com/visionml/pytracking) and [STARK](https://github.com/researchmm/Stark) Library, which helps us to quickly implement our ideas.
