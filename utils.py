from datetime import datetime
import numpy as np

#compute a sinusoidal embedding given a date 
def date_to_sinusoidal_embedding(date_string):
    # Parsing the date string
    date_time_obj = datetime.strptime(date_string.decode("utf-8"), '%Y-%m-%dT%H')
    #print(date_time_obj)
    
    # Extracting components
    #year = (date_time_obj.year - 2000) / 20 # assuming the range of years is from 2000 to 2020
    month = (date_time_obj.month - 1) / 11
    day = (date_time_obj.day - 1) / 30
    hour = date_time_obj.hour / 23
    
    # Creating sinusoidal embeddings
    embedding = []
    for value in [month, day, hour]:
        embedding.append(np.sin(2 * np.pi * value))
        embedding.append(np.cos(2 * np.pi * value))
        
    result = np.array(embedding)
    result = np.tile(result, (96, 16))
    
    return result 

#compute the metrics for two images (ground truth and prediction) 
def compute_metrics(A, B):
    # Check if the input shapes are compatible
    if A.shape != B.shape:
        raise ValueError("Input arrays must have the same shape.")

    # True Positives (TP): Predicted positive and actually positive
    TP = np.sum((A == 1) & (B == 1))

    # True Negatives (TN): Predicted negative and actually negative
    TN = np.sum((A == 0) & (B == 0))

    # False Positives (FP): Predicted positive but actually negative
    FP = np.sum((A == 0) & (B == 1))

    # False Negatives (FN): Predicted negative but actually positive
    FN = np.sum((A == 1) & (B == 0))

    # Accuracy calculation
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    # Precision calculation
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0

    # Recall calculation
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    return accuracy, precision, recall

#compute the threshold 
def threshold_array(arr, x):
    arr[arr > x] = 1
    arr[arr <= x] = 0
    return arr

#compute the metrics for a batch of 3+3 images 
def metrics_aggregator(raw,threshold):
    
    tmp = np.zeros((raw.shape[0],3,3))
    
    for i in range(raw.shape[0]):
    
        hist =raw[i,:,:,:,0:3]
        pred =raw[i,:,:,:,3:]
        
        A = np.copy(hist[:,:,:,-3])
        B = np.copy(pred[:,:,:,-3])

        A = threshold_array(A,threshold)
        B = threshold_array(B,threshold)
        accuracy, precision, recall = compute_metrics(A,B)
        tmp[i,0] = [accuracy, precision, recall]

        A = np.copy(hist[:,:,:,-2])
        B = np.copy(pred[:,:,:,-2])

        A = threshold_array(A,threshold)
        B = threshold_array(B,threshold)
        accuracy, precision, recall = compute_metrics(A,B)
        tmp[i,1] = [accuracy, precision, recall]

        A = np.copy(hist[:,:,:,-1])
        B = np.copy(pred[:,:,:,-1])

        A = threshold_array(A,threshold)
        B = threshold_array(B,threshold)
        accuracy, precision, recall = compute_metrics(A,B)
        tmp[i,2] = [accuracy, precision, recall]

    return tmp
    
