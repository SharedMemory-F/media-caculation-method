%% 读取图像
f = imread('E:\学习\研一\04媒体计算方法\0实验\实验一\Img\Fig0930(a)(calculator).tif') ;
figure;imshow(f);title('原始图像')
%% 01-1 利用imadjust图像灰度变换，改变参数设置
fig1 = figure(1);
set(fig1,'name','实验1-1 灰度处理')
g1 = imadjust(f,[0,1],[1,0],1);
%figure;imshow(g1);title('改变灰度参数1')
g2 = imadjust(f,[0.2,0.5],[0.75,1],1);
%figure;imshow(g2);title('改变灰度参数2');
g3 = imadjust(f,[],[],2);
%figure;imshow(g3);title("改变灰度参数3")
subplot(2,2,1);imshow(f);title('原始图像');
subplot(2,2,2);imshow(g1);title('反转灰度值');
subplot(2,2,3);imshow(g2);title('亮度扩展');
subplot(2,2,4);imshow(g3);title('gamma=2');
%% 01-2 计算函数颜色直方图，并对其进行均衡化处理
fig2 = figure(2);
set(fig2,'name','实验1-2 均衡化处理')
H1 = imhist(f);%原图
h1 = H1(1:10:256);
horz = 1:10:256;
subplot(2,2,2)
stem(horz,h1,'fill')
axis([0 255 0 15000]);
title('原图的直方图')
set(gca,'xtick',[0:50:255]);
set(gca,'ytick',[0:2000:15000]);
g = histeq(f, 16);
H2 = imhist(g);%均衡化的直方图
h2 = H2(1:10:256);
horz = 1:10:256;
subplot(2,2,4)
stem(horz,h2,'fill')
axis([0 255 0 15000]);
title('均衡化的直方图')
set(gca,'xtick',[0:50:255]);
set(gca,'ytick',[0:2000:15000]);
subplot(2,2,1);imshow(f);title('原始图像');
subplot(2,2,3);imshow(g);title('均衡化图像');
%% 02-1 邻域平均法、中值滤波法对图像噪声进行平滑处理并对比
f = imread('E:\学习\研一\04媒体计算方法\0实验\实验一\Img\Fig0409(a)(bld).tif') ;
fig3 = figure(3);
set(fig3,'name','实验2-1 平滑处理')
h1 = imnoise(f, 'gaussian', 0, 0.01);
%figure;imshow(h1);title('高斯噪声')
h2 = imnoise(f, 'salt & pepper');
%figure;imshow(h1);title('椒盐噪声')
m1 = fspecial('average', 3*3);
I1_1 = imfilter(h1,m1);
I1_2 = imfilter(h2,m1);
%figure;imshow(I1_1);title('均值滤波--高斯噪声')
%figure;imshow(I1_2);title('均值滤波--椒盐噪声')
I2_1 = medfilt2(h1);
I2_2 = medfilt2(h2);
%figure;imshow(I2_1);title('中值滤波--高斯噪声')
%figure;imshow(I2_2);title('中值滤波--椒盐噪声')
subplot(2,3,1);imshow(h1);title('高斯噪声图像');
subplot(2,3,4);imshow(h2);title('椒盐噪声图像');
subplot(2,3,2);imshow(I1_1);title('均值滤波--高斯噪声');
subplot(2,3,5);imshow(I1_2);title('均值滤波--椒盐噪声');
subplot(2,3,3);imshow(I2_1);title('中值滤波--高斯噪声');
subplot(2,3,6);imshow(I2_2);title('中值滤波--椒盐噪声');
%% 02-2 用prewitt算子、sobel算子、拉普拉斯算子实现对图像的锐化处理，改变
%% 参数设置，对各算子的锐化效果进行比较
fig4 = figure(4);
set(fig4,'name','实验2-2 锐化处理')
% J1_2改变参数对应K1_2
J1_1 = fspecial('laplacian');%3*3 参数代表算子形状 默认0.2【0，1】
K1_1 = imfilter(f,J1_1);
J2_1 = fspecial('sobel');%边缘提取 无参数
K2_1 = imfilter(f,J2_1);
J3_1 = fspecial('prewitt');%边缘增强，3*3，无参数
K3_1 = imfilter(f,J3_1);
subplot(2,2,1);imshow(f);title('原始图像');
subplot(2,2,2);imshow(K1_1);title('laplacian算子锐化图像');
subplot(2,2,3);imshow(K2_1);title('sobel算子锐化图像');
subplot(2,2,4);imshow(K3_1);title('prewitt算子锐化图像');
%% 03 图像傅里叶变换，显示傅里叶频谱，将频谱中心由原点移到频率矩形的中心。
%% 对变换后的图像进行傅里叶逆变换，比较逆变换后的图像与原图像的差别。
f = imread('E:\学习\研一\04媒体计算方法\0实验\实验一\Img\Fig0409(a)(bld).tif') ;
fig5 = figure(5);
set(fig5,'name','实验3 傅里叶变换')
F_f = fft2(f);
F_fc = fftshift(F_f);
subplot(2,2,1);imshow(f);title('原始图像')
subplot(2,2,2);imshow(F_f,[]);title('傅里叶变换频谱图')
subplot(2,2,3);imshow(log(1+abs(F_fc)), []);title('变换原点移至中心及对数变换')
subplot(2,2,4);imshow(real(ifft2(F_f)),[]);title('傅里叶频谱逆变换原图像')
%figure;imshow(real(ifft2(F_f)));title('傅里叶频谱逆变换原图像') % 原图像
%% 04 构建频域高斯低通滤波器，采用频域滤波的方法，对原图像进行频域滤波处理；将频域滤波的结果与直接在空域进行高斯滤波的效果进行对比。
fig6 = figure(6);
set(fig6,'name','实验4 频域与空域的高斯滤波对比')
[M,N] = size(f);
sig =10;
H = lpfilter('gaussian',M,N,sig);
G = H.*F_f;
g1 = real(ifft2(G));
%空域
K = fspecial('gaussian',500,10);
g2 = imfilter(f,K, 'replicate');
subplot(1,3,1);imshow(f);title('原始图像');
subplot(1,3,2);imshow(g1,[]);title('频域高斯滤波');
subplot(1,3,3);imshow(g2,[]);title('空域高斯滤波');
