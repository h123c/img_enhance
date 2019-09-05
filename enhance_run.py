import os
import argparse
import random
import math
from multiprocessing import Process
from multiprocessing import cpu_count
import data_enhance
from config import params_dict
from cv2 import cv2 as cv

import multiprocessing
multiprocessing.set_start_method('spawn',True)

# 导入image_augmentation.py为一个可调用模块

# 利用Python的argparse模块读取输入输出和各种扰动参数
def parse_args():
    parser = argparse.ArgumentParser(
        description='A Simple Image Data Augmentation Tool',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--input_dir',
                        default=params_dict["input_dir"],
                        help='Directory containing images')
    parser.add_argument('--output_dir',
                        default=params_dict["output_dir"],
                        help='Directory for augmented images')
    parser.add_argument('--num',
                        default=params_dict["num"],
                        help='Number of images to be augmented',
                        type=int)
    parser.add_argument('--noise_num',
                        default=params_dict["noise_num"],
                        help='Number of noise to the image',
                        type=int)

    parser.add_argument('--num_procs',
                        default=params_dict["num_procs"],
                        help='Number of processes for paralleled augmentation',
                        type=int)
    
    parser.add_argument('--Gaussian_core',
                        default=params_dict["Gaussian_core"],
                        help='the core of Gaussian')
    
    parser.add_argument('--Gaussian_std',
                        default=params_dict["Gaussian_std"],
                        help='the std of Gaussian')
    
    parser.add_argument('--transform_index',
                        default=params_dict["transform_index"],
                        help='the index of transform')
                        
    parser.add_argument('--p_blend',
                        default=params_dict["p_blend"],
                        help='the radio of add blend')

    parser.add_argument('--median_blur',
                        default=params_dict["median_blur"],
                        help='the param of median blur')
    parser.add_argument('--p_Gblur',
                        default=params_dict["p_Gblur"],
                        help='the radio of Gaussian blur')

    parser.add_argument('--gauss_noise',
                        default=params_dict["gauss_noise"],
                        help='the params of Gaussian noise')

    parser.add_argument('--kernel_blur',
                        default=params_dict["kernel_blur"],
                        help='the kernel of kernel blur')

    parser.add_argument('--move_array',
                        default=params_dict["move_array"],
                        help='the start index array of moving the image')

    parser.add_argument('--p_mirror',
                        default=params_dict["p_mirror"],
                        help='Ratio to mirror an image',
                        type=float)

    parser.add_argument('--p_noise',
                        default=params_dict["p_noise"],
                        help='Ratio to add noise an image',
                        type=float)

    parser.add_argument('--p_Gnoise',
                        default=params_dict["p_Gnoise"],
                        help='Ratio to add Gaussian noise an image',
                        type=float)

    parser.add_argument('--p_crop',
                        default=params_dict["p_crop"],
                        help='Ratio to randomly crop an image',
                        type=float)

    parser.add_argument('--p_bi',
                        default=params_dict["p_bi"],
                        help='Ratio to bi crop an image',
                        type=float)

    parser.add_argument('--p_sh',
                        default=params_dict["p_sh"],
                        help='Ratio to sh crop an image',
                        type=float)

    parser.add_argument('--p_blur',
                        default=params_dict["p_blur"],
                        help='Ratio to blur crop an image',
                        type=float)

    parser.add_argument('--aver_blur',
                        default=params_dict["aver_blur"],
                        help='weight and height to average blur an image')

    parser.add_argument('--p_mblur',
                        default=params_dict["p_mblur"],
                        help='Ratio to middle blur crop an image',
                        type=float)

    parser.add_argument('--p_trams',
                        default=params_dict["p_trams"],
                        help='Ratio to transform an image',
                        type=float)
    
    parser.add_argument('--p_move',
                        default=params_dict["p_move"],
                        help='Ratio to move an image',
                        type=float)

    parser.add_argument('--p_cblur',
                        default=params_dict["p_cblur"],
                        help='Ratio to cnn blur crop an image',
                        type=float)

    parser.add_argument('--p_zoom',
                        default=params_dict["p_zoom"],
                        help='Ratio to randomly zoom an image',
                        type=float)

    parser.add_argument('--crop_size',
                        default=params_dict["crop_size"],
                        help='The ratio of cropped image size to original image size, in area',
                        type=float)
    parser.add_argument('--crop_hw_vari',
                        default=params_dict["crop_hw_vari"],
                        help='Variation of h/w ratio',
                        type=float)

    parser.add_argument('--p_rotate',
                        default=params_dict["p_rotate"],
                        help='Ratio to randomly rotate an image',
                        type=float)
    parser.add_argument('--p_rotate_crop',
                        default=params_dict["p_rotate_crop"],
                        help='Ratio to crop out the empty part in a rotated image',
                        type=float)
    parser.add_argument('--rotate_angle_vari',
                        default=params_dict["rotate_angle_vari"],
                        help='Variation range of rotate angle',
                        type=float)

    parser.add_argument('--p_hsv',
                        default=params_dict["p_hsv"],
                        help='Ratio to randomly change gamma of an image',
                        type=float)
    parser.add_argument('--hue_vari',
                        default=params_dict["hue_vari"],
                        help='Variation of hue',
                        type=int)
    parser.add_argument('--sat_vari',
                        default=params_dict["sat_vari"],
                        help='Variation of saturation',
                        type=float)
    parser.add_argument('--val_vari',
                        default=params_dict["val_vari"],
                        help='Variation of value',
                        type=float)

    parser.add_argument('--p_gamma',
                        default=params_dict["p_gamma"],
                        help='Ratio to randomly change gamma of an image',
                        type=float)
    parser.add_argument('--gamma_vari',
                        default=params_dict["gamma_vari"],
                        help='Variation of gamma',
                        type=float)

    args = parser.parse_args()
    args.input_dir = args.input_dir.rstrip('/')
    args.output_dir = args.output_dir.rstrip('/')

    return args

'''
根据进程数和要增加的目标图片数，
生成每个进程要处理的文件列表和每个文件要增加的数目
'''
def generate_image_list(args):
    # 获取所有文件名和文件总数
    filenames = os.listdir(args.input_dir)
    num_imgs = len(filenames)

	# 计算平均处理的数目并向下取整
    num_ave_aug = int(math.floor(args.num/num_imgs))
	
	# 剩下的部分不足平均分配到每一个文件，所以做成一个随机幸运列表
	# 对于幸运的文件就多增加一个，凑够指定的数目
    rem = args.num - num_ave_aug*num_imgs
    lucky_seq = [True]*rem + [False]*(num_imgs-rem)
    random.shuffle(lucky_seq)

	# 根据平均分配和幸运表策略，
	# 生成每个文件的全路径和对应要增加的数目并放到一个list里
    img_list = [
        (os.sep.join([args.input_dir, filename]), num_ave_aug+1 if lucky else num_ave_aug)
        for filename, lucky in zip(filenames, lucky_seq)
    ]
	
	# 文件可能大小不一，处理时间也不一样，
	# 所以随机打乱，尽可能保证处理时间均匀
    random.shuffle(img_list)

	# 生成每个进程的文件列表，
	# 尽可能均匀地划分每个进程要处理的数目
    length = float(num_imgs) / float(args.num_procs)
    indices = [int(round(i * length)) for i in range(args.num_procs + 1)]
    res_list = [img_list[indices[i]:indices[i + 1]] for i in range(args.num_procs)]
    return res_list


# 主函数
def main():
    # 获取参数
    args = parse_args()
    params_str = str(args)[10:-1]

	# 如果输出文件夹不存在，则建立文件夹
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    print('Starting image data augmentation for {}\n'
          'with\n{}\n'.format(args.input_dir, params_str))

	# 生成每个进程要处理的列表
    sublists = generate_image_list(args)
    #data_enhance.augment_images([["./input/1.jpg",10]],args)
	# 创建进程
    processes = [Process(target=data_enhance.augment_images, args=(x, args, )) for x in sublists]
	# 并行多进程处理
    for p in processes:
        p.start()

    for p in processes:
        p.join()

    print('\nDone!')

if __name__ == '__main__':
    main()
