class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/home/luohui/SYM/tracking/code/STTrack'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/home/luohui/SYM/tracking/code/STTrack/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/home/luohui/SYM/tracking/code/STTrack/pretrained_networks'
        self.lasot_dir = '/home/luohui/SYM/tracking/code/STTrack/data/lasot'
        self.got10k_dir = '/home/luohui/SYM/tracking/code/STTrack/data/got10k'
        self.lasot_lmdb_dir = '/home/luohui/SYM/tracking/code/STTrack/data/lasot_lmdb'
        self.got10k_lmdb_dir = '/home/luohui/SYM/tracking/code/STTrack/data/got10k_lmdb'
        self.trackingnet_dir = '/home/luohui/SYM/tracking/code/STTrack/data/trackingnet'
        self.trackingnet_lmdb_dir = '/home/luohui/SYM/tracking/code/STTrack/data/trackingnet_lmdb'
        self.coco_dir = '/home/luohui/SYM/tracking/code/STTrack/data/coco'
        self.coco_lmdb_dir = '/home/luohui/SYM/tracking/code/STTrack/data/coco_lmdb'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = '/home/luohui/SYM/tracking/code/STTrack/data/vid'
        self.imagenet_lmdb_dir = '/home/luohui/SYM/tracking/code/STTrack/data/vid_lmdb'
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
