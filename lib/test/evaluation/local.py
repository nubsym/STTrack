from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/home/luohui/SYM/tracking/code/STTrack/data/got10k_lmdb'
    settings.got10k_path = '/home/luohui/SYM/tracking/code/STTrack/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_extension_subset_path = '/home/luohui/SYM/tracking/code/STTrack/data/LaSOT_EXTENSION'
    settings.lasot_lmdb_path = '/home/luohui/SYM/tracking/code/STTrack/data/lasot_lmdb'
    settings.lasot_path = '/home/luohui/SYM/tracking/code/STTrack/data/lasot'
    settings.network_path = '/home/luohui/SYM/tracking/code/STTrack/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/home/luohui/SYM/tracking/code/STTrack/data/nfs'
    settings.otb_path = '/home/luohui/SYM/tracking/code/STTrack/data/OTB2015'
    settings.prj_dir = '/home/luohui/SYM/tracking/code/STTrack'
    settings.result_plot_path = '/home/luohui/SYM/tracking/code/STTrack/test/result_plots'
    settings.results_path = '/home/luohui/SYM/tracking/code/STTrack/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/home/luohui/SYM/tracking/code/STTrack'
    settings.segmentation_path = '/home/luohui/SYM/tracking/code/STTrack/test/segmentation_results'
    settings.tc128_path = '/home/luohui/SYM/tracking/code/STTrack/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/home/luohui/SYM/tracking/code/STTrack/data/TNL2K'
    settings.tpl_path = ''
    settings.trackingnet_path = '/home/luohui/SYM/tracking/code/STTrack/data/trackingNet'
    settings.uav_path = '/home/luohui/SYM/tracking/code/STTrack/data/UAV123'
    settings.vot_path = '/home/luohui/SYM/tracking/code/STTrack/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings

