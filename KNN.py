import pandas as pd
import numpy as np
import operator
import math

# ref: http://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/

class K_nearest_neighbors(object):

    def __init__():
        pass
    
    def load_data(self, raw_data_set, split = 0.66):
        nrows = np.array(raw_data_set).shape[0]
        num_train_set = int( nrows * split)
        random_indexs = np.random.randint(0, nrows, num_train_set)
        remaing_indexs = np.array(set(range(nrows)) - set(random_indexs))
        training_set = np.array(raw_data_set)[random_indexs]
        test_set = np.array(raw_data_set)[remaing_indexs]
        assert training_set.shape[0] + test_set.shape[0] == nrows, "the sum of training set and test set must be the same as input data set."
        return list(training_set), list(test_set)
    
    def euclidean_distance(self, data1, data2, length):
        # data1: 1 x p vector, with p-features
        # data2: 1 x p vector, with p-features
        # legnth: we choose the first length features to calculate euclidean distance.
        
        assert length <= len(data1), "the lenght must be less or equal to the number of features"
        assert length <= len(data2), "the lenght must be less or equal to the number of features"
        d1 = data1[:length]
        d2 = data2[:length]
        return math.sqrt((np.array(d1) - np.array(d2))**2)
        
    def get_neighbors(self, data_example, training_set, k, length = 4):
        # data_example: 1-d array with p-features
        # calculate the distance for all training_array respect to all the training data.
        # and then select the first k-closest data set except itself.
        
        distance_caches = list()
        for i in range(len(training_set)):
            dist_temp = euclidean_distance(data_example, training_set[i], length)
            if dist_temp != 0:
                distance_caches.append([training_set[i], dist_temp])
                distance_caches.sort(key = operator.itemgetter(1))
        k_neighbors = [ distance_caches[i][0] for i in range(k)]
        return k_neighbors



    def predict(self, k_neighbors):
        # use k-closest neighbors to make a prediction which class it should belongs to.
        # the last columns of data describe the class of this data.
        class_collections = {}
        
        for data_row in k_neighbors:
            class_temp = data_row[-1]
            if class_temp in class_collections:
                class_collections[class_temp] += 1
            else:
                class_collections[class_temp] = 1
            
        class_collections = sorted(class_collections.items(), key = operator.itemgetter(1), reverse = True)
        return class_collections[0][0]


    def get_accuracy(self, test_set, prediction_results):
        total_rows = len(test_set)
        count = 0
        for i in range(total_rows):
            test_target = test_set[i][-1]
            if test_target == prediction_results:
                count += 1
        return (100.0 * count)/total_rows
    

    def forecast(self, data_set, k, length = 4, split = 0.66):
        print "loading data ..."
        print "split into training set and test set."
        training_set, test_set = self.load_data(self, raw_data_set, split)

        print "use k-closest neighbors to make a prediction."
        prediction_results = list()
        for i in range(len(test_set)):
            k_neighbors = self.get_neighbors(data_example, training_set, k, length)
            prediction_results.append([self.predict(k_neighbors)])

        accuracy =  self.get_accuracy(test_set, prediction_results):
        print "prediction accuracy: (accuracy)".format(accuracy = accuracy)
