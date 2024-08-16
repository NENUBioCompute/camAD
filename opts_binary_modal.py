import argparse
from pathlib import Path
import inspect
import os


def get_abs_path():
    pass

def parse_opts():
    # frame = inspect.currentframe().f_back
    # caller_file = frame.f_code.co_filename
    # rawabs_path = caller_file

    parser = argparse.ArgumentParser()

    parser.add_argument("--pwd", type=str, default=os.path.abspath(__file__))

    ####################################################################################
    #                                    Main setting                                     #
    ####################################################################################
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--modal", type=str, default='PET_MRI',
                        help='(PET | MRI | PET_MRI | )'
                        )
    parser.add_argument("--task", type=str, default='class')  # 指定输入数据的格式，3D， 2D ['class', 'segment']

    parser.add_argument("--data_path", type=str,
                        default='../sda1/qujing/BinaryDatasets/ADNI1_CN0_AD1_76-94-76/ALL/')
    parser.add_argument("--train_path", type=str,
                        default='../sda1/qujing/ADNI1_PET_MRI_9883/ADNI1_CN0_AD1_76-94-76/train/')
    parser.add_argument("--test_path", type=str,
                        default='../sda1/qujing/ADNI1_PET_MRI_9883/ADNI1_CN0_AD1_76-94-76/test/')

    parser.add_argument("--save_path", type=str, default='../sdb1/qujing/Results/BIBM_Binary/')

    parser.add_argument("--random_seed", type=int, default=59)
    parser.add_argument("--k_fold", type=int, default=10)
    parser.add_argument("--n_iter", type=int, default=1) #iterations
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--iter_t", type=int, default=1)  # iterations

    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--test_batch_size", type=int, default=1)

    parser.add_argument("--metrics", type=list, default=['accuracy', 'precision', 'recall', 'F1-score', 'specificity', 'sensitivity', 'AUC', 'confusion_matrix'])  #
    parser.add_argument("--main_metric", type=str, default='F1-score')
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--mid_dim", type=int, default=512)
    parser.add_argument("--num_init_features", type=int, default=16)

    parser.add_argument('--crossAtt_num_heads',
                        default=4,
                        type=int,
                        help='Head number of vit (4 | 6 | 8 | 12 | )')

    parser.add_argument('--model',
                        default='camAD',
                        type=str,
                        help='(camAD| PT_DCN |'
                             )

    parser.add_argument("--notes", type=str, default='improving',
                        help='status of running purpose (improving | testing | repetition | ... )'
                        )

    ####################################################################################
    #                                  Other  setting                                  #
    ####################################################################################
    parser.add_argument("--lr", type=float, default=1e-4)  # 5e-4
    parser.add_argument("--scheduler_step", type=int, default=1)  # 5e-4
    parser.add_argument("--scheduler_gamma", type=float, default=0.99)  # 5e-4
    parser.add_argument("--scheduler_last_epoch", type=int, default=-1)  # 5e-4
    parser.add_argument("--l1_lambda", type=float, default=1e-4)  # 5e-4

    parser.add_argument('--dropout',
                        default=0.5,
                        type=float)
    parser.add_argument(
        '--criterion',
        default='CrossEtp',
        type=str,
        help=
        '(CrossEtp | FocalLoss | b | c | d | e | ')
    parser.add_argument(
        '--criterion_sign',
        default=False,
        type=str,
        help=
        '(True | False')

    parser.add_argument(
        '--optimizer',
        default='Adam',
        type=str,
        help=
        '(Adam | a | b | c | d | e | ')

    ####################################################################################
    #                                   Model:  Resnet                                 #
    ####################################################################################
    parser.add_argument('--resnet_depth',
                        default=18,
                        type=int,
                        help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument("--inplanes", type=int, default=64,
                        help='Inplanes of resnet (16 | 32 | 64 | 128 | 512)')
    parser.add_argument("--approach", type=str, default='3d')  # 2d, 25d, 3d
    # parser.add_argument("--model", type=str, default='0_ResNet_prepretrain')
    # parser.add_argument("--intp_ch", type=int, default="32")
    parser.add_argument("--is_pool", type=int, default=1)
    parser.add_argument("--isAugment", type=int, default=0)
    parser.add_argument("--sample_ratio", type=float, default=1)
    parser.add_argument("--lambda2", type=float, default=0.0000)

    parser.add_argument("--class_scenario", type=str, default='cn_ad')   # cn_mci_ad, mci_ad, cn_mci
    parser.add_argument('--conv1_t_size',
                        default=7,
                        type=int,
                        help='Kernel size in t dim of conv1.')
    parser.add_argument('--conv1_t_stride',
                        default=1,
                        type=int,
                        help='Stride in t dim of conv1.')
    parser.add_argument('--no_max_pool',
                        action='store_true',
                        help='If true, the max pooling after conv1 is removed.')
    parser.add_argument('--resnet_shortcut',
                        default='B',
                        type=str,
                        help='Shortcut type of resnet (A | B)')
    parser.add_argument(
        '--resnet_widen_factor',
        default=1.0,
        type=float,
        help='The number of feature maps of resnet is multiplied by this value')

    ####################################################################################
    #                                    Model:  VIT                                 #
    ####################################################################################
    parser.add_argument('--vit_depth',
                        default=12,
                        type=int,
                        help='Depth of vit (12 | 16 |  |  | )')
    parser.add_argument("--mask_ratio", type=float, default=0.1)
    parser.add_argument('--vit_emb_dim',
                        default=512,
                        type=int,
                        help='Embedding dim of vit (512 |  |  |  | )')
    parser.add_argument('--vit_num_heads',
                        default=8,
                        type=int,
                        help='Head number of vit (4 | 6 | 8 | 12 | )')
    parser.add_argument('--patch_size_2d',
                        default=(4, 4),
                        type=set)
    parser.add_argument('--patch_size_3d',
                        default=(8, 8, 8),
                        type=set)
    parser.add_argument('--vit_in_channels',
                        default=1,
                        type=int)

    ####################################################################################
    #                                   Model: transformer                             #
    ####################################################################################
    parser.add_argument("--d_f", type=int, default=64)

    parser.add_argument("--max_slicelen", type=int, default=114)
    parser.add_argument("--axial_slicelen", type=int, default=96)
    parser.add_argument("--coronal_slicelen", type=int, default=114)

    parser.add_argument("--d_ff", type=int, default=128)
    parser.add_argument("--tf_num_stack", type=int, default=1)
    parser.add_argument("--tf_num_heads", type=int, default=4)


    args = parser.parse_args()

    return args