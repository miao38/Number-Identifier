#Roy Miao
#PUDI: 0029600702
#Project KNN

import csv
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys

#should probably use numpy
class MyKNN:
    #reading in the data set
    class image:
        def __init__(self, label, x, y):
            self.label = label
            self.x = x
            self.y = y

    def __init__(self):
        self.empyt = ""

    def readData (self, filename, number = []):
        try:
            data = {}
            x = 0
            file = csv.reader(open(filename, 'r')) #reading in the csv file
            for i in file:
                if len(number) == 0:
                    data[x] = self.image(int(i[1]), float(i[2]), float(i[3])) #the first position is the label, then the x, and then the y value
                    x += 1
                else:
                    for j in number:
                        if j == int(i[1]):
                            data[x] = self.image(int(i[1]), float(i[2]), float(i[3])) #the first position is the label, then the x, and then the y value
                            x += 1
                            break
        except:
            print("Something went wrong with the readData function")
        return data

    #Gives the predictions
    def classify(self, k, parsedData, trainingIds, testIds):
        try:
            traningData = {}
            prediction = []
            for i in trainingIds:
                traningData[i] = parsedData[i] #setting the key of training data to what the id is
            for i in testIds:
                testData = parsedData[i]
                predictValue = self.prediction(k, testData, traningData) #made a prediction function that calculates what we think the number is
                prediction.append(predictValue)
        except:
            e = sys.exc_info()[0]
            print(e)
            print("Something went wrong with the classify function")
        return prediction

    def prediction (self, k, testData, trainingData):
        try:
            d = {}
            for i in trainingData.keys(): #looping with the training data keys
                distance = math.sqrt(((testData.x - trainingData[i].x)**2) + ((testData.y - trainingData[i].y)**2)) #the distance formula
                d[i] = distance #storing the distance in the dictionary d
            x = 0
            shortest = []
            while (x < k): #trying to find the shortest distance so we know what number it should be
                key = d.keys()[0]
                s = d[d.keys()[0]]
                for i in d.keys():
                    if (s > d[i]):
                        s = d[i]
                        key = i
                shortest.append(key)
                del d[key] #after finding the shortest one, deleting that key so we can find the next shortest one
                x += 1
            count = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}
            for i in shortest:
                number = trainingData[i].label
                count[number] += 1
            largest = 0
            predictedValue = 0
            for i in count.keys():
                if count[i] > largest:
                    largest = count[i]
                    predictedValue = i
        except:
            print("Something went wrong with the prediction function")
        return predictedValue

    #gives the F1 score and the confusion matrix
    def evaluate (self, parsedData, testIds, predictions):
        try:
            confusionMat = np.zeros([10, 10]) #creating a matrix with the index place as the number
            for i in range(len(testIds)):
                if predictions[i] == parsedData[testIds[i]].label:
                    confusionMat[predictions[i]][predictions[i]] += 1 #adding one to the tp place
                else:
                    confusionMat[predictions[i]][parsedData[testIds[i]].label] += 1 #adding 1 to the fp place
            f1List = []
            avgF1 = 0
            for i in range(10): #finding how many tp, fn, and fp values there are, goes through the rows
                tp = 0
                fn = 0
                fp = 0
                for j in range(10): #goes through the columns
                    if i == j: #the tp
                        tp = confusionMat[i][j]
                    elif i != j:
                        fp = fp + confusionMat[i][j]
                for k in range(10): #looping through the column to find the fn
                    if k == i:
                        continue
                    else:
                        fn = fn + confusionMat[k][i]
                if 2*tp+fn != 0:
                    f1List.append((2*tp)/(2*tp+fp+fn))
            for i in f1List:
                avgF1 = avgF1 + i
            avgF1 = avgF1/len(f1List)
        except:
            print("Something went wrong with the evaluate function")
        return avgF1, confusionMat





knn = MyKNN()
parsedData = knn.readData('digits-embedding.csv', [1, 2])
trainingIds = []
for i in range(len(parsedData)):
    if i % 2 == 0:
        trainingIds.append(i)
testingIds = []
for i in range(len(parsedData)):
    if i % 2 != 0:
        testingIds.append(i)
#prediction = knn.classify(5, parsedData, trainingIds, testingIds)
#knn.evaluate(parsedData, testingIds, prediction)
#Reporting Tasks
#i.1.
xList = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[]}
yList = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[]}
for i in parsedData.keys():
    image = parsedData[i]
    xList[image.label].append(image.x)
    yList[image.label].append(image.y)
colors = ['red', 'green', 'blue', 'purple', 'black', 'yellow', 'orange', 'cyan', 'magenta', 'pink']
label = ['0','1','2','3','4','5','6','7','8','9']
zero = []
for i in range(len(label)):
    zero.append(mpatches.Patch(color = colors[i], label = label[i]))
for i in range(len(xList)):
    plt.scatter(xList[i], yList[i], color = colors[i], label = "2D Features")
plt.legend(handles = zero)
plt.title("2D Feature")
plt.show()
#2 and 3
k = [1, 5, 15, 31] #was told not to include q from piazza
fList = []
for i in k:
    prediction = knn.classify(i, parsedData, trainingIds, testingIds)
    f1Score, matrix = knn.evaluate(parsedData, testingIds, prediction)
    fList.append(f1Score)
    print("For k = " + str(i))
    print("F1 Score: " + str(f1Score))
    print("Matrix: ")
    print(matrix)
    print()
#4.
plt.plot(k, fList)
plt.title("Average F1 Score vs k")
plt.xlabel("k")
plt.ylabel("Average F1 Score")
plt.show()