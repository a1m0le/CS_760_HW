import numpy as np


INIT_STEP_SIZE = 0.1
MAX_STEPS = 100


class datapoint:

    def __init__(self, point, label):
        self.point = point
        self.label = label



#TODO: add regularization(lambda)???
class myLogiClassifier:

    def __init__(self, filename):
        self.filename = filename
        self.test_range = None
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
        #init theta
        self.theta_len = len(self.overall_data[0].point)
        self.theta = None

    

    def set_test_range(self, lo, hi):
        self.test_range = (lo, hi)


    def compute_gradient(self, dp):
        #TODO
        x = dp.point
        y = dp.label
        # firstly, compute dot product
        theta_T_x = np.dot(self.theta, x)
        # now the sigmoid
        sigmoid = np.exp(theta_T_x) / (1 + np.exp(theta_T_x))
        # now the inside
        inside = sigmoid - y
        # now the gradient
        gradient = inside * x
        return gradient


    
    def step_tune(self, original):
        return original
        #new_step = original * 0.5
        #return new_step



    def train(self):
        #TODO
        # reset theta
        self.theta = np.zeros(self.theta_len)
        # do this many steps:
        past_gradient = None
        step = INIT_STEP_SIZE
        for step in range(0, MAX_STEPS):
            # start gradient descent
            gradient_sum = np.zeros(self.theta_len)
            count = 0
            for j in range(0, len(self.overall_data)):
                if j >= self.test_range[0] and j <self.test_range[1]:
                    continue # Skip test points
                dp = self.overall_data[j]
                single_grad = self.compute_gradient(dp)
                gradient_sum = gradient_sum + single_grad
                count = count + 1
            avg_gradient = gradient_sum / count
            if past_gradient is None:
                past_gradient = avg_gradient
            else:
                #TODO: tunning??
                pass
            movement = step * avg_gradient
            self.theta = self.thta - movement
        

    def predict(self):
        predictions = []
        actuals = []
        for i in range(test_range[0], test_range[1]):
            test_dp = self.overall_data[i]
            test_pt = test_dp.point
            test_label = test_dp.label
            # do prediction
            theta_T_x = np.dot(self.theta, test_pt)
            sigmoid = np.exp(theta_T_x) / (1 + np.exp(theta_T_x))
            predict_label = None
            if sigmoid >= 0.5:
                predict_label = 1
            else:
                predict_label = 0
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
        return accuracy, precision, recall







