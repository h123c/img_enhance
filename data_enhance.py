#encoding:utf-8
#导入库
from cv2 import cv2 as cv
import numpy as np
import random,os
def zoom_img(img,radio_tuple):
    #缩放
    height,width = img.shape[:2] #获取图片的高和宽
    #将图像缩小为原来的0.5倍
    img_zoom = cv.resize(img, (int(width*radio_tuple[0]),int(height*radio_tuple[1])), interpolation=cv.INTER_CUBIC)
    # cv.imshow('zoom', img_zoom) #显示图片
    # cv.waitKey(0)  
    # cv.destroyAllWindows()
    return img_zoom

def bright_contrast1(img,contrast = 1,brightness = 100):
    #亮度与对比度contrast:对比度 brightness:亮度
    pic_turn = cv.addWeighted(img,contrast,img,0,brightness)
    #cv.addWeighted(对象,对比度,对象,对比度)
    '''cv.addWeighted()实现的是图像透明度的改变与图像的叠加'''
    # cv.imshow('turn', pic_turn) #显示图片
    # cv.waitKey(0)  
    # cv.destroyAllWindows()
    return pic_turn

def bright_contrast2( img_path , contrast , brightness):
    pic = cv.imread(img_path)
    """
    f(x)= contrast * g(x) + brightness
    通过遍历图像的高度，宽度，通道数分别去改变它们对应的值
    pic.shape[0]:图像中像素行的数量或图像阵列的每列中的像素数。
    pic.shape[1]:图像中像素列的数量或图像阵列的每行中的像素数。
    pic.shape[2]:用于表示每个像素的组件数。
    """
    for i in range(pic.shape[0]):
        for j in range(pic.shape[1]):
            for k in range(pic.shape[2]):
                if (pic[i][j][k] * contrast+ brightness<=255):
                    pic[i][j][k] = pic[i][j][k] * contrast+ brightness
                else:
                    pic[i][j][k] = 255
    return pic


def blend_img(img1):
    #混合,调整透明度
    img2 = np.ones(img1.shape,dtype=np.uint8)
    #img = np.random.random((3,3)) #生成随机数都是小数无法转化颜色,无法调用cv2.cvtColor函数
    img2[:,:,0]=random.randint(0,255)
    img2[:,:,1]=random.randint(0,255)
    img2[:,:,2]=random.randint(0,255)
    val = random.uniform(0.05,0.3)
    diaphaneity=(1-val,val)   #(0.7,0.3)
    dst = cv.addWeighted(img1, diaphaneity[0],img2, diaphaneity[1],0)
    # cv.imshow('dst',dst)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    return dst
def Flip_img(img):
    """
    翻转
    flip(img,1)#1代表水平方向旋转180度
    flip(img,0)#0代表垂直方向旋转180度
    flip(img,-1)#-1代表垂直和水平方向同时旋转
    """
    pic = cv.imread(img_path) #读入图片

    h_pic = cv.flip(pic, 1)#水平翻转
    cv.imshow("overturn-h", h_pic)

    v_pic = cv.flip(pic, 0)#垂直翻转
    cv.imshow("overturn-v", v_pic)

    hv_pic = cv.flip(pic, -1)#水平垂直翻转
    cv.imshow("overturn-hv", hv_pic)
    cv.waitKey(0)
    cv.destroyAllWindows()

def rotate(img, angle, scale=1.0):
    #旋转
    height,width = img.shape[:2] #获取图像的高和宽
    center = (width / 2, height / 2) #取图像的中点

    M = cv.getRotationMatrix2D(center, angle, scale)#获得图像绕着某一点的旋转矩阵 
    rotated = cv.warpAffine(img, M, (height, width))
    #cv.warpAffine()的第二个参数是变换矩阵,第三个参数是输出图像的大小
    # cv.imshow("overturn-hv", rotated)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    return rotated

'''
定义旋转函数：
angle是逆时针旋转的角度
crop是个布尔值，表明是否要裁剪去除黑边
'''
def rotate_image(img, angle, crop):
    h, w = img.shape[:2]
	
	# 旋转角度的周期是360°
    angle %= 360
	
	# 用OpenCV内置函数计算仿射矩阵
    M_rotate = cv.getRotationMatrix2D((w/2, h/2), angle, 1)
	
	# 得到旋转后的图像
    img_rotated = cv.warpAffine(img, M_rotate, (w, h))

	# 如果需要裁剪去除黑边
    if crop:
        angle_crop = angle % 180             	    # 对于裁剪角度的等效周期是180°
        if angle_crop > 90:                        	# 并且关于90°对称
            angle_crop = 180 - angle_crop
		
        theta = angle_crop * np.pi / 180.0    		# 转化角度为弧度
        hw_ratio = float(h) / float(w)    		    # 计算高宽比
		

        tan_theta = np.tan(theta)                   # 计算裁剪边长系数的分子项
        numerator = np.cos(theta) + np.sin(theta) * tan_theta
		

        r = hw_ratio if h > w else 1 / hw_ratio		# 计算分母项中和宽高比相关的项
        denominator = r * tan_theta + 1		 		# 计算分母项

        crop_mult = numerator / denominator			# 计算最终的边长系数
        w_crop = int(round(crop_mult*w))			# 得到裁剪区域
        h_crop = int(round(crop_mult*h))
        x0 = int((w-w_crop)/2)
        y0 = int((h-h_crop)/2)
        img_rotated = crop_image(img_rotated, x0, y0, w_crop, h_crop)
    return img_rotated


def random_rotate(img, angle_vari, p_crop):
    '''
    随机旋转
    angle_vari是旋转角度的范围[-angle_vari, angle_vari)
    p_crop是要进行去黑边裁剪的比例
    '''
    angle = np.random.uniform(-angle_vari, angle_vari)
    crop = False if np.random.random() > p_crop else True
    return rotate_image(img, angle, crop)


def add_noise(img,noise_num=1000):
    #添加噪声
    for i in range(noise_num):
        img[random.randint(0, img.shape[0]-1)][random.randint(0,img.shape[1]-1)][:]=255
    # cv.imshow('pic_noise', img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    return img

def Gaussian_blur(img,Gaussian_core,Gaussian_std):
    #高斯模糊
    temp = cv.GaussianBlur(img, Gaussian_core, Gaussian_std)
    #      cv.GaussianBlur(图像，卷积核，标准差）
    # cv.imshow("img_blur", temp)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    return temp

def move_img(img,move_array=[[1,0,-100],[0,1,-12]]):
    #平移
    #平移矩阵[[1,0,-100],[0,1,-12]]
    M=np.array(move_array,dtype=np.float32)
    img_change=cv.warpAffine(img,M,(300,300))
    # cv.imshow("test",img_change)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    return img_change

#定义高斯噪声
def gaussian_noise(image,gauss_noise):
    h,w,ch = image.shape
    variance = random.randint(1,gauss_noise[1])
    gauss_noise[1] = variance
    for row in range(h):
        for col in range(w):
            s = np.random.normal(gauss_noise)   #均值为0 方差为20 输出3个值
            b = image[row,col,0]    #blue
            g = image[row,col,1]    #green
            r = image[row,col,2]    #red
            # 给blue层加上正态分布噪点
            image[row,col,0] = b + s[0]
            image[row,col,1] = g + s[0]
            image[row,col,2] = r + s[0]
    #cv.imshow('noise image',image)
    return image
#clamp 夹紧在范围内

"""
#EPF-高斯双边滤波
其中各参数所表达的意义：
    src：原图像；
    d：像素的邻域直径，可有sigmaColor和sigmaSpace计算可得；
    sigmaColor：颜色空间的标准方差，一般尽可能大；
    sigmaSpace：坐标空间的标准方差(像素单位)，一般尽可能小。
"""
def bi_demo(image):
    dst = cv.bilateralFilter(image,0,150,10)
    #cv.imshow('bilateralFilter',dst)
    return dst


"""
#EPF-均值偏移滤波
其中各参数所表达的意义：
    src：原图像;
    sp：空间窗的半径(The spatial window radius);
    sr：色彩窗的半径(The color window radius);
注意: 通过均值迁移来进行边缘保留滤波有时会导致图像过度模糊
"""
def shift_demo(image):
    dst = cv.pyrMeanShiftFiltering(image,10,50)
    #cv.imshow('pyrMeanShiftFiltering',dst)
    return dst

def transform_img(img,plane1,plane2):
    # 仿射
    # 对图像进行变换（三点得到一个变换矩阵）
    # 我们知道三点确定一个平面，我们也可以通过确定三个点的关系来得到转换矩阵
    # 然后再通过warpAffine来进行变换
    rows,cols=img.shape[:2]

    point1=np.float32(plane1)
    point2=np.float32(plane2)
    #point2=np.float32([[10,100],[300,50],[100,250]])

    M=cv.getAffineTransform(point1,point2)
    dst=cv.warpAffine(img,M,(cols,rows),borderValue=(255,255,255))

    # cv.imshow("1",dst)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    return dst

'''
定义hsv变换函数：
hue_delta是色调变化比例
sat_delta是饱和度变化比例
val_delta是明度变化比例
'''
def hsv_transform(img, hue_delta, sat_mult, val_mult):
    img_hsv = cv.cvtColor(img.astype(np.uint8), cv.COLOR_BGR2HSV)#.astype()
    img_hsv[:, :, 0] = hue_delta*img_hsv[:, :, 0]#(img_hsv[:, :, 0] + hue_delta) % 180
    img_hsv[:, :, 1] = sat_mult*img_hsv[:, :, 1]
    img_hsv[:, :, 2] = val_mult*img_hsv[:, :, 2]
    img_hsv[img_hsv > 255] = 255
    dst = cv.cvtColor(np.round(img_hsv).astype(np.uint8), cv.COLOR_HSV2BGR)
    # cv.imshow("1",dst)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    return dst

'''
随机hsv变换
hue_vari是色调变化比例的范围
sat_vari是饱和度变化比例的范围
val_vari是明度变化比例的范围
'''
def random_hsv_transform(img, hue_vari, sat_vari, val_vari):
    hue_delta = 0.9 + np.random.uniform(-hue_vari, hue_vari)
    sat_mult = 0.9 + np.random.uniform(-sat_vari, sat_vari)
    val_mult = 0.9 + np.random.uniform(-val_vari, val_vari)
    return hsv_transform(img, hue_delta, sat_mult, val_mult)

'''
定义gamma变换函数：
gamma就是Gamma
'''
def gamma_transform(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    dst = cv.LUT(img, gamma_table)
    # cv.imshow("1",dst)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    return dst

'''
随机gamma变换
gamma_vari是Gamma变化的范围[1/gamma_vari, gamma_vari)
'''
def random_gamma_transform(img, gamma_vari):
    log_gamma_vari = np.log(gamma_vari)
    alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)
    gamma = np.exp(alpha)
    return gamma_transform(img, gamma)


crop_image = lambda img, x0, y0, w, h: img[y0:y0+h, x0:x0+w]
'''
定义裁剪函数，四个参数分别是：
左上角横坐标x0
左上角纵坐标y0
裁剪宽度w
裁剪高度h
'''

def random_crop(img, area_ratio, hw_vari):
    '''
    随机裁剪
    area_ratio为裁剪画面占原画面的比例
    hw_vari是扰动占原高宽比的比例范围
    '''
    
    h, w = img.shape[:2]
    #area = h*w
    hw_delta = np.random.uniform(-hw_vari, hw_vari)
    hw_mult = 1 + hw_delta
	
	# 下标进行裁剪，宽高必须是正整数
    w_crop = int(round(w*np.sqrt(area_ratio*hw_mult)))
	
	# 裁剪宽度不可超过原图可裁剪宽度
    if w_crop > w:
        w_crop = w
		
    h_crop = int(round(h*np.sqrt(area_ratio*hw_mult)))
    if h_crop > h:
        h_crop = h
	# 随机生成左上角的位置
    x0 = np.random.randint(0, w-w_crop+1)
    y0 = np.random.randint(0, h-h_crop+1)
    
    dst = crop_image(img, x0, y0, w_crop, h_crop)
    return dst 

#均值模糊 (2,8)高模糊2 宽模糊8
def blur_demo(image,aver_blur=(4,4)):
    h = np.random.randint(1, aver_blur[0])
    w = np.random.randint(1, aver_blur[1])
    dst = cv.blur(image,(w,h))
    #cv.imshow('blur_demo',dst)
    return dst

#中值模糊 适合椒盐噪声去噪
def median_blur_demo(image,median_blur=3):
    dst = cv.medianBlur(image,median_blur)
    #cv.imshow('median_blur_demo',dst)
    return dst

#自定义模糊
def customer_blur_demo(image,kernel_blur=[[0,-1,0],[-1,5,-1],[0,-1,0]]):
    #定义卷积核---均值模糊的效果
    # kernel = np.ones([5,5],np.float32/25)
    # 定义卷积核---锐化
    kernel = np.array(kernel_blur,np.float32)

    dst = cv.filter2D(image,-1,kernel=kernel)
    #cv.imshow('customer_blur_demo',dst)
    return dst

def augment_images(filelist, args):
    # 遍历所有列表内的文件
    for filepath, n in filelist:
        img = cv.imread(filepath)
        filename = filepath.split(os.sep)[-1]
        dot_pos = filename.rfind('.')
		
		# 获取文件名和后缀名
        imgname = filename[:dot_pos]
        ext = filename[dot_pos:]

        print('Augmenting {} ...'.format(filename))
        for i in range(n):
            img_varied = img.copy()
			
			# 扰动后文件名的前缀
            varied_imgname = '{}_{:0>3d}_'.format(imgname, i)
			
            # 1.按照比例随机对图像进行 缩放
            if random.random() < args.p_zoom:
                radio_w = random.uniform(0.5,1.5)
                radio_h = random.uniform(0.5,1.5)
                img_varied = zoom_img(img_varied, (radio_w,radio_h))
                varied_imgname += '_z'

			# 2.按照比例随机对图像进行 镜像
            if random.random() < args.p_mirror:
                flage = random.randint(-1,1)
                img_varied = cv.flip(img_varied, flage)
                varied_imgname += '_m'
			
			# 3.按照比例随机对图像进行 裁剪
            if random.random() < args.p_crop:
                img_varied = random_crop(
                    img_varied,
                    args.crop_size,
                    args.crop_hw_vari)
                varied_imgname += '_c'
			
			# 4.按照比例随机对图像进行 旋转
            if random.random() < args.p_rotate:
                img_varied = random_rotate(
                    img_varied,
                    args.rotate_angle_vari,
                    args.p_rotate_crop)
                varied_imgname += '_r'
            
            # 5.按照比例随机对图像进行 加噪
            if random.random() < args.p_noise:
                img_varied = add_noise(
                    img_varied,
                    args.noise_num)
                varied_imgname += '_n'

            # 6.按照比例随机对图像进行 加高斯模糊
            if random.random() < args.p_Gblur:
                size_list=[i for i in range(3,args.Gaussian_core,2)]
                ind = random.randint(0,len(size_list)-1)
                core_std = random.uniform(0.3,args.Gaussian_std)
                core_tuple = (size_list[ind],size_list[ind])
                img_varied = Gaussian_blur(
                    img_varied,
                    core_tuple,
                    core_std)
                varied_imgname += '_Gb'
            
            # 7.按照比例随机对图像进行 平移
            if random.random() < args.p_move:
                img_varied = move_img(
                    img_varied,
                    args.move_array)
                varied_imgname += '_mv'

            # 8.按照比例随机对图像进行 仿射
            if random.random() < args.p_trams:
                pix_sum = img_varied.sum()
                new_sum = pix_sum*2
                while new_sum > (1.0*pix_sum) or new_sum < (0.5*pix_sum):
                    index = np.array([i for i in range(0,360,5)])
                    plane1 = []
                    plane2 = []
                    for i in range(3):
                        ind1 = random.randint(0,71)
                        ind2 = random.randint(0,71)
                        plane1.append([index[ind1],index[ind2]])
                        size1 = random.randint(0,4)
                        size2 = random.randint(0,4)
                        if ind1+size1 > 71:new_ind1 = 71
                        elif ind1+size1 < 0:new_ind1 = 0
                        else:new_ind1 = ind1+size1
                        
                        if ind2+size2 > 71:new_ind2 = 71
                        elif ind2+size2 < 0:new_ind2 = 0
                        else:new_ind2 = ind2+size2
                        plane2.append([index[new_ind1],index[new_ind2]])
                    
                    # for i in range(3):
                    #     ind1 = random.randint(0,71)
                    #     ind2 = random.randint(0,71)
                    #     plane2.append([index[ind1],index[ind2]])
                    # plane1 = [index[i] for i in args.transform_index[0]]
                    # plane2 = [index[i] for i in args.transform_index[1]]
                    new_img_varied = transform_img(
                        img_varied,
                        plane1,plane2)
                    new_sum = new_img_varied.sum()
                img_varied = new_img_varied
                varied_imgname += '_t'
			
			# 9.按照比例随机对图像进行 HSV扰动
            if random.random() < args.p_hsv:
                img_varied = random_hsv_transform(
                    img_varied,
                    args.hue_vari,
                    args.sat_vari,
                    args.val_vari)
                varied_imgname += '_h'
			
			# 10.按照比例随机对图像进行 Gamma扰动
            if random.random() < args.p_gamma:
                img_varied = random_gamma_transform(
                    img_varied,
                    args.gamma_vari)
                varied_imgname += '_g'

            # 11.按照比例随机对图像进行 均值模糊
            if random.random() < args.p_blur:
                img_varied = blur_demo(
                    img_varied,
                    args.aver_blur)
                varied_imgname += '_b'
            
            # 12.按照比例随机对图像进行 中值模糊
            if random.random() < args.p_mblur:
                img_varied = median_blur_demo(
                    img_varied,
                    args.median_blur)
                varied_imgname += '_mb'

            # 13.按照比例随机对图像进行 卷积模糊
            if random.random() < args.p_cblur:
                img_varied = customer_blur_demo(
                    img_varied,
                    args.kernel_blur)
                varied_imgname += '_cb'
            
            # 14.按照比例随机对图像进行 高斯双边滤波
            if random.random() < args.p_bi:
                img_varied = bi_demo(
                    img_varied)
                varied_imgname += '_bi'
            
            # 15.按照比例随机对图像进行 均值偏移滤波
            if random.random() < args.p_sh:
                img_varied = shift_demo(
                    img_varied)
                varied_imgname += '_sh'
            
            # 16.按照比例随机对图像进行 高斯噪声
            if random.random() < args.p_Gnoise:
                img_varied = gaussian_noise(
                    img_varied,
                    args.gauss_noise)
                varied_imgname += '_Gn'
            
            # 17.按照比例随机对图像进行 添加透明纯色背景
            if random.random() < args.p_blend:
                img_varied = blend_img(
                    img_varied)
                varied_imgname += '_bl'
            print(varied_imgname)
			# 生成扰动后的文件名并保存在指定的路径
            output_filepath = os.sep.join([
                args.output_dir,
                '{}{}'.format(varied_imgname, ext)])
            cv.imwrite(output_filepath, img_varied)

if __name__ == "__main__":
    img_path = "1.jpg"
    
    # radio = (0.5,0.3)
    # zoom_img(img_path,radio) #缩放
    
    #bright_contrast(img_path)  #亮度 对比度

    # img_path2 = "2.jpg"
    # blend_img(img_path,img_path2) #混合,透明度

    #Flip_img(img_path)  #翻转

    # angle = 45
    # rotate(img_path, angle) #旋转

    #add_noise(img_path) #噪声

    #Gaussian_noise(img_path) #高斯模糊

    #move_img(img_path) #平移

    #transform_img(img_path) #仿射

    gamma_vari = 30
    random_gamma_transform(img_path, gamma_vari)

"""
参考:
https://blog.csdn.net/weixin_42730096/article/details/84439805
https://blog.csdn.net/wsp_1138886114/article/details/83028948
2、data augmentation常用方法

    Color Jittering：对颜色的数据增强：图像亮度、饱和度、对比度变化（此处对色彩抖动的理解不知是否得当）；
    PCA  Jittering：首先按照RGB三个颜色通道计算均值和标准差，再在整个训练集上计算协方差矩阵，进行特征分解，得到特征向量和特征值，用来做PCA Jittering；
    Random Scale：尺度变换；
    Random Crop：采用随机图像差值方式，对图像进行裁剪、缩放；包括Scale Jittering方法（VGG及ResNet模型使用）或者尺度和长宽比增强变换；
    Horizontal/Vertical Flip：水平/垂直翻转；
    Shift：平移变换；
    Rotation/Reflection：旋转/仿射变换；
    Noise：高斯噪声、模糊处理；
    Label shuffle：类别不平衡数据的增广，
https://blog.csdn.net/u010555688/article/details/60757932
"""