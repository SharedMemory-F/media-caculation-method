%% ��ȡͼ��
f = imread('E:\ѧϰ\��һ\04ý����㷽��\0ʵ��\ʵ��һ\Img\Fig0930(a)(calculator).tif') ;
figure;imshow(f);title('ԭʼͼ��')
%% 01-1 ����imadjustͼ��Ҷȱ任���ı��������
fig1 = figure(1);
set(fig1,'name','ʵ��1-1 �Ҷȴ���')
g1 = imadjust(f,[0,1],[1,0],1);
%figure;imshow(g1);title('�ı�ҶȲ���1')
g2 = imadjust(f,[0.2,0.5],[0.75,1],1);
%figure;imshow(g2);title('�ı�ҶȲ���2');
g3 = imadjust(f,[],[],2);
%figure;imshow(g3);title("�ı�ҶȲ���3")
subplot(2,2,1);imshow(f);title('ԭʼͼ��');
subplot(2,2,2);imshow(g1);title('��ת�Ҷ�ֵ');
subplot(2,2,3);imshow(g2);title('������չ');
subplot(2,2,4);imshow(g3);title('gamma=2');
%% 01-2 ���㺯����ɫֱ��ͼ����������о��⻯����
fig2 = figure(2);
set(fig2,'name','ʵ��1-2 ���⻯����')
H1 = imhist(f);%ԭͼ
h1 = H1(1:10:256);
horz = 1:10:256;
subplot(2,2,2)
stem(horz,h1,'fill')
axis([0 255 0 15000]);
title('ԭͼ��ֱ��ͼ')
set(gca,'xtick',[0:50:255]);
set(gca,'ytick',[0:2000:15000]);
g = histeq(f, 16);
H2 = imhist(g);%���⻯��ֱ��ͼ
h2 = H2(1:10:256);
horz = 1:10:256;
subplot(2,2,4)
stem(horz,h2,'fill')
axis([0 255 0 15000]);
title('���⻯��ֱ��ͼ')
set(gca,'xtick',[0:50:255]);
set(gca,'ytick',[0:2000:15000]);
subplot(2,2,1);imshow(f);title('ԭʼͼ��');
subplot(2,2,3);imshow(g);title('���⻯ͼ��');
%% 02-1 ����ƽ��������ֵ�˲�����ͼ����������ƽ�������Ա�
f = imread('E:\ѧϰ\��һ\04ý����㷽��\0ʵ��\ʵ��һ\Img\Fig0409(a)(bld).tif') ;
fig3 = figure(3);
set(fig3,'name','ʵ��2-1 ƽ������')
h1 = imnoise(f, 'gaussian', 0, 0.01);
%figure;imshow(h1);title('��˹����')
h2 = imnoise(f, 'salt & pepper');
%figure;imshow(h1);title('��������')
m1 = fspecial('average', 3*3);
I1_1 = imfilter(h1,m1);
I1_2 = imfilter(h2,m1);
%figure;imshow(I1_1);title('��ֵ�˲�--��˹����')
%figure;imshow(I1_2);title('��ֵ�˲�--��������')
I2_1 = medfilt2(h1);
I2_2 = medfilt2(h2);
%figure;imshow(I2_1);title('��ֵ�˲�--��˹����')
%figure;imshow(I2_2);title('��ֵ�˲�--��������')
subplot(2,3,1);imshow(h1);title('��˹����ͼ��');
subplot(2,3,4);imshow(h2);title('��������ͼ��');
subplot(2,3,2);imshow(I1_1);title('��ֵ�˲�--��˹����');
subplot(2,3,5);imshow(I1_2);title('��ֵ�˲�--��������');
subplot(2,3,3);imshow(I2_1);title('��ֵ�˲�--��˹����');
subplot(2,3,6);imshow(I2_2);title('��ֵ�˲�--��������');
%% 02-2 ��prewitt���ӡ�sobel���ӡ�������˹����ʵ�ֶ�ͼ����񻯴����ı�
%% �������ã��Ը����ӵ���Ч�����бȽ�
fig4 = figure(4);
set(fig4,'name','ʵ��2-2 �񻯴���')
% J1_2�ı������ӦK1_2
J1_1 = fspecial('laplacian');%3*3 ��������������״ Ĭ��0.2��0��1��
K1_1 = imfilter(f,J1_1);
J2_1 = fspecial('sobel');%��Ե��ȡ �޲���
K2_1 = imfilter(f,J2_1);
J3_1 = fspecial('prewitt');%��Ե��ǿ��3*3���޲���
K3_1 = imfilter(f,J3_1);
subplot(2,2,1);imshow(f);title('ԭʼͼ��');
subplot(2,2,2);imshow(K1_1);title('laplacian������ͼ��');
subplot(2,2,3);imshow(K2_1);title('sobel������ͼ��');
subplot(2,2,4);imshow(K3_1);title('prewitt������ͼ��');
%% 03 ͼ����Ҷ�任����ʾ����ҶƵ�ף���Ƶ��������ԭ���Ƶ�Ƶ�ʾ��ε����ġ�
%% �Ա任���ͼ����и���Ҷ��任���Ƚ���任���ͼ����ԭͼ��Ĳ��
f = imread('E:\ѧϰ\��һ\04ý����㷽��\0ʵ��\ʵ��һ\Img\Fig0409(a)(bld).tif') ;
fig5 = figure(5);
set(fig5,'name','ʵ��3 ����Ҷ�任')
F_f = fft2(f);
F_fc = fftshift(F_f);
subplot(2,2,1);imshow(f);title('ԭʼͼ��')
subplot(2,2,2);imshow(F_f,[]);title('����Ҷ�任Ƶ��ͼ')
subplot(2,2,3);imshow(log(1+abs(F_fc)), []);title('�任ԭ���������ļ������任')
subplot(2,2,4);imshow(real(ifft2(F_f)),[]);title('����ҶƵ����任ԭͼ��')
%figure;imshow(real(ifft2(F_f)));title('����ҶƵ����任ԭͼ��') % ԭͼ��
%% 04 ����Ƶ���˹��ͨ�˲���������Ƶ���˲��ķ�������ԭͼ�����Ƶ���˲�������Ƶ���˲��Ľ����ֱ���ڿ�����и�˹�˲���Ч�����жԱȡ�
fig6 = figure(6);
set(fig6,'name','ʵ��4 Ƶ�������ĸ�˹�˲��Ա�')
[M,N] = size(f);
sig =10;
H = lpfilter('gaussian',M,N,sig);
G = H.*F_f;
g1 = real(ifft2(G));
%����
K = fspecial('gaussian',500,10);
g2 = imfilter(f,K, 'replicate');
subplot(1,3,1);imshow(f);title('ԭʼͼ��');
subplot(1,3,2);imshow(g1,[]);title('Ƶ���˹�˲�');
subplot(1,3,3);imshow(g2,[]);title('�����˹�˲�');
