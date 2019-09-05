#enocidng:utf-8
from multiprocessing import cpu_count

params_dict={
	"input_dir":"./input/",
	"output_dir":"./output/",
	"num":20,			#增强后总数量
	"num_procs":4,	#进程数 cpu_count()
	"p_zoom":0.2,	#缩放百分比
	"p_mirror":0.2,	#镜像百分比
	"p_crop":0.2,	#裁danyi剪百分比
	"p_rotate":0.2,	#旋danyi转百分比
	"p_noise":0.4,	#加danyi噪声百分比
	"p_Gnoise":0.05,	#加高斯噪声百分比 太单一,0.05
	"p_Gblur":0.1,	#加高斯模糊百分比
	"p_move":0.0,	#平移百分比      有问题!!!
	"p_trams":0.2,	#仿射百分比      
	"p_hsv":0.2,	#HSV扰动百分比
	"p_gamma":0.2,	#gamma扰动百分比
	"p_blur":0.2,	#均值模糊百分比
	"p_mblur":0.2,	#中值模糊百分比 椒盐
	"p_cblur":0.2,	#卷积模糊百分比
	"p_bi":0.2,		#高斯双边滤波百分比
	"p_sh":0.2,		#均值偏移滤波百分比
	"p_blend":0.2,		#添加纯色背景百分比
	"p_rotate_crop":1.0, #去黑边裁剪的比例
	"rotate_angle_vari":30.0, #旋转角度的范围
	"zoom_radio":(0.5,0.5),				#缩放尺寸比例
	"contrast":1,						#亮度
	"brightness":100,					#对比度
	"crop_size":0.6,						#裁剪
	"crop_hw_vari":0.15,
	"diaphaneity":(0.7,0.3),			#透明度 不能改动
	"angle":360,						#逆时针旋转角度
	"crop":True,						#是否裁剪旋转的黑边
	"noise_num":1000,					#噪声点
	"Gaussian_core":7,				#高斯模糊卷积核
	"Gaussian_std":2,					#高斯模糊标准差
	"move_array":[[1,0,-100],[0,1,-12]],#平移矩阵
	"gauss_noise":[0,10,3],				#高斯噪声
	"transform_index":([[3,5],[15,11],[56,42]],[[67,61],[46,50],[24,36]]), #仿射变化坐标 [[50,50],[300,50],[50,200]],[[10,100],[300,50],[100,250]]
	"hue_vari":0.1,						#色调变化比例
	"sat_vari":0.1,       				#饱和度变化比例
	"val_vari":0.1,						#明度变化比例
	"gamma_vari":100.0,					#是Gamma变化的范围
	"aver_blur":(4,6), 					#高模糊,宽模糊
	"median_blur":3,					#椒盐模糊   必须是奇数
	"kernel_blur":[[0,-1,0],[-1,5,-1],[0,-1,0]],#自定义卷积模糊
}
"""
缩放
zoom_img(img,radio)
亮度,对比度
bright_contrast1(img,contrast = 1,brightness = 100)
混合
blend_img(img1,img2, diaphaneity=(0.7,0.3))
随机旋转
random_rotate(img, angle_vari, p_crop)
添加噪声
add_noise(img,noise_num=1000)
高斯噪声
Gaussian_noise(img,Gaussian_core=(7,7),Gaussian_std=1.5)
平移
move_img(img,move_array=[[1,0,-100],[0,1,-12]])
高斯噪声
gaussian_noise(image,gauss_noise=(0,20,3))
高斯双边滤波
bi_demo(image)
均值偏移滤波
shift_demo(image)
仿射
transform_img(img,transform_index)
随机hsv变换
random_hsv_transform(img, hue_vari, sat_vari, val_vari)
随机gamma函数变化
random_gamma_transform(img, gamma_vari)
随机裁剪
random_crop(img, area_ratio, hw_vari)
均值模糊
blur_demo(image,aver_blur=(2,8))
中值模糊
median_blur_demo(image,median_blur=5)
自定义卷积模糊
customer_blur_demo(image,kernel_blur=[[0,-1,0],[-1,5,-1],[0,-1,0]])


"""