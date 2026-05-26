import argparse

def parse_opts():
    parser = argparse.ArgumentParser()

    ''' Overall Settings '''
    parser.add_argument('--root_path', type=str, default='')
    parser.add_argument('--data_path', type=str, default='')
    #parser.add_argument('--LDM', type=str, default='CcLDM', choices=['CcLDM'])
    parser.add_argument('--seed', type=int, default=2020, metavar='S', help='random seed (default: 2020)')


    ''' Dataset '''
    ## Data split: fluid dataset is split into a train set (first 800 time points in each block) and a test set (the last 200 points); the unique labels in two sets do not overlap.
    parser.add_argument('--data_split', type=str, default='train',
                        choices=['all', 'train'])
    parser.add_argument('--min_label', type=float, default=0.0)
    parser.add_argument('--max_label', type=float, default=2.1)
    parser.add_argument('--num_channels', type=int, default=3, metavar='N')
    #parser.add_argument('--img_size', type=int, default=64, metavar='N',
    #                    choices=[64])
    parser.add_argument('--max_num_img_per_label', type=int, default=50, metavar='N')
    parser.add_argument('--max_num_img_per_label_after_replica', type=int, default=0, metavar='N')
    parser.add_argument('--visualize_fake_images', action='store_true', default=False)


    ''' DM settings '''
    parser.add_argument('--niters', type=int, default=10000, help='number of iterations')
    parser.add_argument('--resume_niters', type=int, default=0)
    parser.add_argument('--save_niters_freq', type=int, default=2000, help='frequency of saving checkpoints')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate for model')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--cLDM_num_classes', type=int, default=90, metavar='N') #bin label into cLDM_num_classes

    parser.add_argument('--kernel_sigma', type=float, default=-1.0,
                        help='If kernel_sigma<0, then use rule-of-thumb formula to compute the sigma.')
    parser.add_argument('--threshold_type', type=str, default='hard', choices=['soft', 'hard'])
    parser.add_argument('--kappa', type=float, default=-1)
    parser.add_argument('--nonzero_soft_weight_threshold', type=float, default=1e-3,
                        help='threshold for determining nonzero weights for SVDL; we neglect images with too small weights')

    # evaluation setting
    '''
    Four evaluation modes:
    Mode 1: eval on unique labels used for DM training;
    Mode 2. eval on all unique labels in the dataset and when computing FID use all real images in the dataset (default);
    Mode 3. eval on all unique labels in the dataset and when computing FID only use real images for DM training in the dataset (to test SFID's effectiveness on unseen labels);
    Mode 4. eval on a interval [min_label, max_label] with num_eval_labels labels.
    '''
    parser.add_argument('--eval_mode', type=int, default=2)
    parser.add_argument('--num_eval_labels', type=int, default=-1)
    parser.add_argument('--samp_batch_size', type=int, default=1000)
    parser.add_argument('--nfake_per_label', type=int, default=50)
    parser.add_argument('--nreal_per_label', type=int, default=-1)
    parser.add_argument('--comp_FID', action='store_true', default=False)
    parser.add_argument('--epoch_FID_CNN', type=int, default=200)
    parser.add_argument('--FID_radius', type=float, default=0)
    parser.add_argument('--FID_num_centers', type=int, default=-1)
    parser.add_argument('--FID_CNN_feature', type=str, default='f3', choices=['f2', 'f3', 'f4'])
    parser.add_argument('--dump_fake_for_NIQE', action='store_true', default=False)
    parser.add_argument('--comp_IS_and_FID_only', action='store_true', default=False)

    args = parser.parse_args()

    return args

