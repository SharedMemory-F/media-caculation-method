import cv2 as cv
from matplotlib import pyplot as plt

print("test opencv start")

image = cv.imread("014_0001.jpg")

channels = cv.split(image)
colors = ('b', 'g', 'r')

hist = cv.calcHist([image], [0], None, [256], [0, 256])
plt.figure()  # 新建一个图像
plt.title("RGB Histogram")  # 图像的标题
plt.xlabel("Bins")  # X轴标签
plt.ylabel("# of Pixels")  # Y轴标签

for (channels, color) in zip(channels, colors):
    hist = cv.calcHist([channels], [0], None, [256], [0, 256])
    plt.plot(hist, color=color)
    plt.xlim([0, 256])
#plt.show()  # 显示图像
plt.savefig('014_0001_rgb_histogram.jpg')

cv.waitKey(0)
cv.destroyAllWindows()

print("test opencv end.")