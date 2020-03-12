import numpy as np
import cv2

def video2frames(video_path):
    #将视频转化为帧，video_path为视频路径，如果路径为空时则打开电脑摄像头
    capture=cv2.VideoCapture()
    #打开视频文件
    capture.open(video_path)
    frames=[]
    #逐帧读取视频文件
    while True:
        #ret为bool值，frame为视频帧，当ret为False时退出
        ret,frame=capture.read()
        if ret:
            frames.append(frame)
        else:
            break
    return frames

if __name__ == "__main__":
    #读入视频，转化为帧
    frames=video2frames("Stuff/Multiple_View.avi")
    #展示视频内容
    '''
    for frame in frames:
        cv2.imshow("frame",frame)
        cv2.waitKey(1)
    '''
    #读取参考图片
    ref_img=cv2.imread("Stuff/ReferenceFrame.png")
    #图片的高，宽
    h,w=ref_img.shape[0],ref_img.shape[1]

    #读取AU层目标
    augment_img=cv2.imread("Stuff/AugmentedLayer.PNG")
    augment_img=augment_img[:h,:w]
    #读取AU层mask
    augment_mask_img=cv2.imread("Stuff/AugmentedLayerMask.PNG")
    augment_mask_img=augment_mask_img[:h,:w]
    augment_mask_img=cv2.bitwise_not(augment_mask_img)
    #将AU层的目标与参考帧融合
    au_ref_img=cv2.bitwise_and(ref_img,augment_mask_img)
    au_ref_img=cv2.bitwise_or(au_ref_img,augment_img)

    cv2.imshow("AU_reference_img",au_ref_img)
    cv2.waitKey(0)
    cv2.destroyWindow("AU_reference_img")
    #读取mask图片
    mask_object=cv2.imread("Stuff/ObjectMask.PNG")
    mask_object=mask_object[:h,:w]
    ref_img=cv2.bitwise_and(ref_img,mask_object)

    #创建特征点检测函数,使用ORB方法
    detect=cv2.ORB_create()
    #从参考帧中检测关键点
    ref_kp=detect.detect(ref_img,None)
    ref_kp,ref_des=detect.compute(ref_img,ref_kp)
    #创建一个空图像
    ref_kp_img=np.zeros_like(ref_img)
    #绘制参考帧上的关键点
    cv2.drawKeypoints(ref_img, ref_kp,ref_kp_img, color=(0,255,0))
    cv2.imshow('Detected keypoints', ref_kp_img)
    cv2.waitKey(0)
    cv2.destroyWindow('Detected keypoints')
    #创建Brute Force算法匹配器
    bf=cv2.BFMatcher_create()
    #逐帧检测特征
    for frame in frames:
        try:
            #检测每一帧的特征点和描述符
            frame_kp=detect.detect(frame,None)
            frame_kp,frame_des=detect.compute(frame,frame_kp)
            #将每一帧的特征点与参考帧的特征点进行匹配
            matches=bf.knnMatch(ref_des,frame_des,k=2)
            #选择距离较近的匹配
            matchesMask = [[0,0] for i in range(len(matches))]
            match_points=[]
            for i,(m,n) in enumerate(matches):
                if m.distance < 0.7*n.distance:
                    matchesMask[i]=[1,0]
                    match_points.append(m)
            #将特征点的匹配关系进行绘制
            draw_params = dict(matchColor = (0,255,0),
                    singlePointColor = (255,0,0),
                    matchesMask = matchesMask,
                    flags = 0)
            frame_match = cv2.drawMatchesKnn(ref_img,ref_kp,frame,frame_kp,matches,None,**draw_params)

            #获取关键点的坐标，一个匹配包含query和train两部分，分别取这两部分的index
            ref_pts = np.float32([ ref_kp[m.queryIdx].pt for m in match_points ]).reshape(-1,1,2)
            frame_pts = np.float32([ frame_kp[m.trainIdx].pt for m in match_points ]).reshape(-1,1,2)
            #使用RANSAC算法计算单应性矩阵
            H, mask = cv2.findHomography(ref_pts,frame_pts, cv2.RANSAC,5.0)
            #旋转目标mask,从帧中提取目标
            frame_obj_mask=cv2.warpAffine(mask_object,H[:2,:],dsize=(frame.shape[1],frame.shape[0]))
            frame_masked=cv2.bitwise_and(frame,frame_obj_mask)
            frame_masked_not=cv2.bitwise_and(frame,cv2.bitwise_not(frame_obj_mask))
            #旋转AU层mask
            frame_au_mask=cv2.warpAffine(augment_mask_img,H[:2,:],dsize=(frame.shape[1],frame.shape[0]))
            #旋转AU层
            frame_au=cv2.warpAffine(augment_img,H[:2,:],dsize=(frame.shape[1],frame.shape[0]))
            #将AU层放置到帧上
            au_frame_img=cv2.bitwise_and(frame_masked,frame_au_mask)
            au_frame_img=cv2.bitwise_or(au_frame_img,frame_au)
            au_frame_img=cv2.bitwise_or(au_frame_img,frame_masked_not)

            cv2.imshow("frame",frame)
            cv2.imshow("object mask",au_frame_img)
            cv2.imshow("frame_match",frame_match)
            cv2.waitKey(2)
        except:
            continue
