import cv2 
import numpy as np
import os

def getSetInfo(dataset, dataset_path):
    #dataset = {}
    subdir = os.listdir(dataset_path)
    for subdir_name in subdir:
        file_path = dataset_path+os.sep+subdir_name+os.sep
        for _ in os.listdir(file_path):
            # print(file_name)
            dataset.setdefault(subdir_name, 0)
            dataset[subdir_name] += 1

def calcSiftFeature(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create(250)
    kp, des = sift.detectAndCompute(gray, None)
    img = cv2.drawKeypoints(image=img, outImage=img, keypoints=kp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,color=(0, 0, 255))
    cv2.imwrite('sift_feature.jpg',img)
    return des

def calcFeatVec(features, centers):
	featVec = np.zeros((1, 50))
	for i in range(0, features.shape[0]):
		fi = features[i]
		diffMat = np.tile(fi, (50, 1)) - centers
		sqSum = (diffMat**2).sum(axis=1)
		dist = sqSum**0.5
		sortedIndices = dist.argsort()
		idx = sortedIndices[0] # index of the nearest center
		featVec[0][idx] += 1	
	return featVec

def initFeatureSet():
    TrainSetInfo = {}
    getSetInfo(TrainSetInfo, "data/train")
    for name, count in TrainSetInfo.items():
        dir = "data/train"+os.sep+ name + os.sep
        featureSet = np.float32([]).reshape(0,128)

        print("Extract features from training set " + name + "...")
        pathDir = os.listdir(dir)
        for i in pathDir:
            filename = os.path.join('%s%s' % (dir, i))
            img = cv2.imread(filename)
            des = calcSiftFeature(img)
            featureSet = np.append(featureSet, des, axis=0)
            featCnt = featureSet.shape[0]
        print(str(featCnt) + " features in " + str(count) + " images\n")

		# save featureSet to file
        filename = "data/Temp1/features/" + name + ".npy"
        np.save(filename, featureSet)

def learnVocabulary():
    TrainSetInfo={}
    getSetInfo(TrainSetInfo, "data/train")
    wordCnt = 50
    for name, count in TrainSetInfo.items():
        filename = "data/Temp1/features/" + name + ".npy"
        features = np.load(filename)
        print("Learn vocabulary of " + name + "...")
		# use k-means to cluster a bag of features
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.1)
        flags = cv2.KMEANS_RANDOM_CENTERS
        compactness, labels, centers = cv2.kmeans(features, wordCnt,None, criteria, 20, flags)
		# save vocabulary(a tuple of (labels, centers)) to file
        filename = "data/Temp1/vocabulary/" + name + ".npy"
        np.save(filename, (labels, centers))
        print("Done\n")

def trainClassifier():
    TrainSetInfo = {}
    getSetInfo(TrainSetInfo, "data/train")
    trainData = np.float32([]).reshape(0, 50)
    response = np.int32([])
    dictIdx = 0
    for name, count in TrainSetInfo.items():
        dir = "data/train/" + name + os.sep
        labels, centers = np.load("data/Temp1/vocabulary/" + name + ".npy")
		
        print("Init training data of " + name + "...")
        pathDir = os.listdir(dir)
        for i in pathDir:
            filename = os.path.join('%s%s' % (dir, i))
            img = cv2.imread(filename)
            features = calcSiftFeature(img)
            featVec = calcFeatVec(features, centers)
            trainData = np.append(trainData, featVec, axis=0)
		
        res = np.repeat(dictIdx, count)
        response = np.append(response, res)
        dictIdx += 1
        print("Done\n")
    print("Now train svm classifier...")
    trainData = np.float32(trainData)
    response = response.reshape(-1, 1)
    svm = cv2.ml.SVM_create()
    svm.setKernel(cv2.ml.SVM_LINEAR)
    # svm.train_auto(trainData, response, None, None, None) # select best params
    svm.train(trainData,cv2.ml.ROW_SAMPLE,response)
    svm.save("svm.clf")
    print("Done\n")

def classify():
	# svm = cv2.SVM()
	# svm = cv2.ml.SVM_create()
    # TrainSetInfo = {}
    # getSetInfo(TrainSetInfo, "data/train")
    TestSetInfo = {}
    getSetInfo(TestSetInfo, "data/val")
    svm = cv2.ml.SVM_load("svm.clf")

    total = 0; correct = 0; dictIdx = 0
    for name, count in TestSetInfo.items():
        crt = 0

        dir = "data/val/" + name + "/"
        labels, centers = np.load("data/Temp1/vocabulary/" + name + ".npy")

        print("Classify on TestSet " + name + ":")
        pathDir = os.listdir(dir)
        for i in pathDir:
            filename = os.path.join('%s%s' % (dir, i))
            img = cv2.imread(filename)
            features = calcSiftFeature(img)
            featVec = calcFeatVec(features, centers)
            case = np.float32(featVec)
            dict_svm = svm.predict(case)
            dict_svm = int(dict_svm[1])
            if (dictIdx == dict_svm):
                crt += 1
        print("Accuracy: " + str(crt) + " / " + str(count) + "\n")
        total += count
        correct += crt
        dictIdx += 1

    print("Total accuracy: " + str(correct) + " / " + str(total) + '---' + str(round(correct/total,5)) )
 
if __name__ == "__main__":
    # initFeatureSet()
    # learnVocabulary()
    trainClassifier()
    # img = cv2.imread("014_0001.jpg")
    # calcSiftFeature(img)
    # classify()
    # TrainSetInfo = {}
    # TestSetInfo = {}

    # getSetInfo(TrainSetInfo, "data/train") #获取TrainSetInfo
    # getSetInfo(TestSetInfo, "data/val") #获取TestSetInfo
    