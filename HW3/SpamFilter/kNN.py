import numpy as np
import heapq


#class too_simple_neighbors_bucket
#TODO: do this if necessary



class neighbor_bucket:


    # decides how many neighbors allowed
    def __init__(self, k):
        self.k = k
        self.data = []
        heapq.heapify(self.data)
    
    def add(self, dist, label):
        neg_dist = dist * -1 # for min heap purpose
        bundle = (neg_dist, label)
        if len(self.data) < self.k:
            # we can still add
            heapq.heappush(self.data, bundle)
        else:
            # check for the top of the queue
            neg_furthest = self.data[0][0]
            if neg_furthest > neg_dist:
                return # stop. no need
            # we can add it
            heapq.heappushpop(self.data, bundle) # should be able to add in and kick out


    def popular_label(self):
        label_counts = {}
        for d in self.data:
            if d[1] not in label_counts:
                label_counts[d[1]] = 0
            label_counts[d[1]] = label_counts[d[1]] + 1
        max_count = 0
        popular = None
        for key in label_counts:
            if label_counts[key] > max_count:
                max_count = label_counts[key]
                popular = key
        # 50:50 case handling
        if max_count == self.k/2:
            return 0
        return popular


class datapoint:

    def __init__(self, point, label):
        self.point = point
        self.label = label

class myKNN:


    def __init__(self, filename):
        self.filename = filename
        self.k = 1
        # process the csv file
        self.overall_data = []
        allines = None
        with open(self.filename) as f:
            allines = f.readlines()
        for i in range(1, len(allines)):
            dataline = allines[i]
            datapart = dataline.split(" ")[1]
            data = datapart.split(",")
            label = int(data[-1])
            pt_floats = []
            for j in range(1, len(data)-1):
                pt_floats.append(float(data[j]))
            point = np.array(pt_floats)
            dp = datapoint(point, label)
            self.overall_data.append(dp)
            


    def config_k(self, k):
        self.k = k
        

    # given training range and test range, return prediction results in the form of (accuracy, precision, recall)
    # ranges are (inclusive index, exclusive upperbound)
    def predict(self, test_range):
        predictions = []
        actuals = []
        for i in range(test_range[0], test_range[1]):
            test_dp = self.overall_data[i]
            test_pt = test_dp.point
            test_label = test_dp.label
            knn_bucket = neighbor_bucket(self.k)
            for j in range(0, len(self.overall_data)):
                if j >= test_range[0] and j <test_range[1]:
                    continue
                train_dp = self.overall_data[j]
                train_pt = train_dp.point
                train_label = train_dp.label
                dist = np.linalg.norm(test_pt - train_pt)
                knn_bucket.add(dist, train_label)
            predict_label = knn_bucket.popular_label()
            predictions.append(predict_label)
            actuals.append(test_label)
        # now we have all our predictions we can start calculations
        TP = 0
        FP = 0
        FN = 0
        TN = 0
        for i in range(0, len(predictions)):
            predict = predictions[i]
            actual = actuals[i]
            if predict == 1 and actual == 1:
                TP = TP + 1
            elif predict == 1 and actual == 0:
                FP = FP + 1
            elif predict == 0 and actual == 1:
                FN = FN + 1
            elif predict == 0 and actual == 0:
                TN = TN + 1
            else:
                raise Exception("How can we have p="+str(predict)+"  a="+str(actual)+" ??????")
        accuracy = (TP + TN) / (TP + FP + FN + TN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        return accuracy, precision, recall, predictions, actuals


    # this one is for determining ROC
    def predict_ROC(self, test_range):
        confidence_data = []
        for i in range(test_range[0], test_range[1]):
            test_dp = self.overall_data[i]
            test_pt = test_dp.point
            test_label = test_dp.label
            knn_bucket = neighbor_bucket(self.k)
            for j in range(0, len(self.overall_data)):
                if j >= test_range[0] and j <test_range[1]:
                    continue
                train_dp = self.overall_data[j]
                train_pt = train_dp.point
                train_label = train_dp.label
                dist = np.linalg.norm(test_pt - train_pt)
                knn_bucket.add(dist, train_label)

            neighbors = knn_bucket.data
            ones = 0
            for n in neighbors:
                if n[1] == 1:
                    ones = ones + 1
            confidence = ones / 5
            confi_entry = (confidence, test_label)
            confidence_data.append(confi_entry)
        # now we have all our predictions we can start calculations
        confidence_data.sort(key=lambda entry:entry[0], reverse=True)
        return confidence_data
        






