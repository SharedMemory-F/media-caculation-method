import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2


class LBP:
    def __init__(self):
        return

        # 读入图片，并且转化为灰度图，获取图像灰度图的像素信息
    def describe(self, image_name):
        image = cv2.imread(image_name)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image_array = np.array(image_gray)
        #image_array = np.array(Image.open(image).convert('L'))
        return image_array

    def calulate_origin_lbp(self, image_array, i, j):
        '''
        原始的特征计算算法：将图像指定位置的像素与周围八个像素比较，比中心像素
        大的点赋值为1，比中心像素小的赋值为0，返回二进制序列
        '''
        sum = []
        if image_array[i-1][j-1] > image_array[i][j]:
            sum.append(1)
        else:
            sum.append(0)
        if image_array[i][j-1] > image_array[i][j]:
            sum.append(1)
        else:
            sum.append(0)
        if image_array[i+1][j-1] > image_array[i][j]:
            sum.append(1)
        else:
            sum.append(0)
        if image_array[i+1][j] > image_array[i][j]:
            sum.append(1)
        else:
            sum.append(0)
        if image_array[i+1][j+1] > image_array[i][j]:
            sum.append(1)
        else:
            sum.append(0)
        if image_array[i][j+1] > image_array[i][j]:
            sum.append(1)
        else:
            sum.append(0)
        if image_array[i-1][j+1] > image_array[i][j]:
            sum.append(1)
        else:
            sum.append(0)
        if image_array[i-1][j] > image_array[i][j]:
            sum.append(1)
        else:
            sum.append(0)
        return sum

    def feature_origin_lbp(self, image_array):
        '''利用原始特征计算方法获取特征'''
        origin_array = np.zeros(image_array.shape, np.uint8)
        width = image_array.shape[0]
        height = image_array.shape[1]
        for i in range(1, width-1):
            for j in range(1, height-1):
                sum = self.calulate_origin_lbp(image_array, i, j)
                bit_num = 0
                result = 0
                for signal in sum:
                    result += signal << bit_num
                    bit_num += 1
                origin_array[i][j] = result
        return origin_array

    def calculate_r(self, r):
        '''计算跳r的变次数'''
        num = 0
        while(r):
            r &= (r-1)
            num += 1
        return num

    def get_uniform_map(self,):
        '''获取等价模式的58中特征从小到大进行序列化编号得到的字典'''
        values = []
        # 等价模式的特征二进制表示中1都是连续出现的
        # 模拟连续出现1-7个1，然后求其旋转一周的八个值
        for i in range(1, 8):
            init_value = 0
            init_length = 8
            arr = [init_value]*init_length
            j = 0
            while(j < i):
                arr[j] = 1
                j += 1
            # 二进制表示旋转一周的八个值
            for i in range(0, 8):
                b = 0
                sum = 0
                for j in range(i, 8):
                    sum += arr[j] << b
                    b += 1
                for k in range(0, i):
                    sum += arr[k] << b
                    b += 1
                values.append(sum)
        values.append(0)
        values.append(255)
        values.sort()
        num = 0
        uniform_map = {}
        for v in values:
            uniform_map[v] = num
            num += 1
        return uniform_map

    def feature_uniform_lbp(self, image_array):
        '''利用等价模式获取特征'''
        uniform_array = np.zeros(image_array.shape, np.uint8)
        width = image_array.shape[0]
        height = image_array.shape[1]
        origin_array = self.feature_origin_lbp(image_array)
        uniform_map = self.get_uniform_map()

        for i in range(width-1):
            for j in range(height-1):
                k = origin_array[i, j] << 1
                if k > 255:
                    k = k-255
                xor = origin_array[i, j] ^ k
                num = self.calculate_r(xor)
                if num <= 2:
                    uniform_array[i, j] = uniform_map[origin_array[i, j]]
                else:
                    uniform_array[i, j] = 58
        return uniform_array

    def get_min_for_resolve(self, image_array):
        # 获取二进制序列进行不断环形旋转得到新的二进制序列的最小十进制
        #result = 0
        values = []
        circle = image_array
        circle.extend(image_array)
        for i in range(0,8):
            j = 0
            sum = 0
            bit_num = 0
            while j < 8:
                sum += circle[i+j] << bit_num
                bit_num += 1
                j += 1
            values.append(sum)
        return min(values)

    # 获取旋转字典模板
    def get_resolve_map(self, image_array):
        values = []
        for i in range(0,256):
            b = bin(i)
            length = len(b)
            arr = []
            j = length - 1
            while(j>1):
                arr.append(int(b[j]))
                j -= 1
            for _ in range(0,8-len(arr)):
                arr.append(0)
            h = []
            circle = arr
            circle.extend(arr)
            for m in range(0,8):
                k = 0
                sum = 0
                bit_num = 0
                while k<8:
                    sum += circle[m+k]<<bit_num
                    bit_num += 1
                    k +=1
                h.append(sum)
            values.append(min(h))
        values.sort()
        resolve_map = {}
        num = 0
        for v in values:
            if v not in resolve_map.keys():
                resolve_map[v] = num 
                num += 1
        return resolve_map

    # 获取图像的旋转不变特征
    def feature_resolve_lbp(self, image_array):
        resolve_array = np.zeros(image_array.shape, np.uint8)
        width = image_array.shape[0]
        height = image_array.shape[1]
        resolve_map = self.get_resolve_map(image_array)
        for i in range(1, width-1):
            for j in range(1, height-1):
                sum = self.calulate_origin_lbp(image_array, i, j)
                resolve_key = self.get_min_for_resolve(sum)
                resolve_array[i,j] = resolve_map[resolve_key]
        return resolve_array
    # def show_imge(self, image_array):
    #     plt.imshow(image_array)
    #     plt.show()

    def show_hist(self, image_array, im_bins, im_range):
        hist = cv2.calcHist([image_array], [0], None, im_bins, im_range)
        hist = cv2.normalize(hist, None).flatten()
        plt.plot(hist, color='r')
        plt.xlim(im_range)
        # plt.show()

    def show_result(self, image_array,origin_feature,uniform_feature,resolve_feature):
        plt.figure(1)
        plt.subplot(321)
        plt.imshow(origin_feature)
        plt.subplot(322)
        self.show_hist(origin_feature, [256], [0, 256])
        plt.subplot(323)
        plt.imshow(uniform_feature)
        plt.subplot(324)
        self.show_hist(uniform_feature, [256], [0, 256])
        plt.subplot(325)
        plt.imshow(resolve_feature)
        plt.subplot(326)
        self.show_hist(resolve_feature, [256], [0, 256])
        plt.show()


if __name__ == '__main__':
    image_name = 'origin.png'
    lbp = LBP()
    image_array = lbp.describe(image_name)  # 灰度图
    origin_feature_array = lbp.feature_origin_lbp(image_array)  # 原始方法求特征
    # lbp.show_origin_hist(origin_feature_array)
    # # lbp = LBP()
    # print(lbp.get_uniform_map())
    # # 获取图像的等价模式LBP特征
    
    uniform_feature_array = lbp.feature_uniform_lbp(image_array)
    # #lbp.show_origin_hist(uniform_feature_array)
    # print(lbp.get_resolve_map(image_array))
    resolve_feature_array = lbp.feature_uniform_lbp(image_array)
    print(resolve_feature_array)
    lbp.show_result(image_array, origin_feature_array, uniform_feature_array, resolve_feature_array)
    # # lbp.show_origin_hist(resolve_feature_array)
    # # print(origin_feature_array)
    # # lbp.show_origin_hist(image_array)#画出直方图
