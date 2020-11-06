#!/usr/bin/env python
# coding: utf-8

import os
import statistics
from handout import data

def train_RF():
    for i in range(10):
        embeddingPath = base_path+"fold_"+str(i)+"/datasetEmbeddingsWithVecPseTrain.pkl"
        saveModelPath = base_path+"fold_"+str(i)+"/trainedRFModel.pkl"
        evaluateEmbeddings = base_path+"fold_"+str(i)+"/datasetEmbeddingsWithVecPseTest.pkl"
        os.system("python source.py --model RF --train True --embeddingPath {0} --save True "
              "--saveModelPath {1} --evaluateEmbeddings {2} ".format(embeddingPath, saveModelPath, evaluateEmbeddings))

def train_lstm():
    for i in range(10):
        embeddingPath = base_path+"fold_"+str(i)+"/datasetEmbeddingsWithTensorPseTrain.pkl"
        saveModelPath = base_path+"fold_"+str(i)+"/trainedSABiLSTMModel.pkl"
        evaluateEmbeddings = base_path+"fold_"+str(i)+"/datasetEmbeddingsWithTensorPseDev.pkl"
        os.system("python source.py --model SABiLSTM  --embeddingPath {0} --train True --save True "
              "--saveModelPath {1} --evaluateEmbeddings {2} --memoryGrowth True --plot True".format(embeddingPath, saveModelPath, evaluateEmbeddings))

def train_ensemble():
    for i in range(10):
        embeddingPath = base_path + "fold_" + str(i) + "/datasetEmbeddingsWithVecPseTrain.pkl"
        saveModelPath = base_path+"fold_"+str(i)+"/trainedSABiLSTMModel.pkl"
        evaluateEmbeddingsVec = base_path + "fold_" + str(i) + "/datasetEmbeddingsWithVecPseTest.pkl"
        evaluateEmbeddings = base_path + "fold_" + str(i) + "/datasetEmbeddingsWithTensorPseTest.pkl"
        exit_code = os.system("python source.py --model Ensemble  --embeddingPath {0} --save True "
                  "--saveModelPath {1} --evaluateEmbeddings {2} --evaluateEmbeddingsVec {3}".format(embeddingPath, saveModelPath,
                                                                         evaluateEmbeddings, evaluateEmbeddingsVec))
        if exit_code == 1:
            exit(1)

# generate embedding tensors and vectors for 10 folds
data.genTenorFor10Folds()
data.genVecFor10Folds()

# set base path
base_path = "handout/pseudomonas/train_cv/"

#clean statistics file
with open("handout/pseudomonas/train_cv/statistics.txt", "w") as f:
         pass

#choose the model to train
train_ensemble()

#evaluations and statistics
with open("handout/pseudomonas/train_cv/statistics.txt","r") as f:
    ACCs, ROCAUCs, PRCAUCs = [],[],[]
    fps,fns = [],[]
    for line in f.readlines():
        line = line.split(" ")
        ACCs.append(float(line[0]))
        ROCAUCs.append(float(line[1]))
        PRCAUCs.append(float(line[2]))
        fps.append(int(line[3]))
        fns.append(int(line[4]))
    print("10 folds average:")
    #print("ACC:", statistics.mean(ACCs))
    print('ROC-AUC: ', statistics.mean(ROCAUCs), '+/-', statistics.pstdev(ROCAUCs), sep='')
    print('PRC-AUC: ', statistics.mean(PRCAUCs), '+/-', statistics.pstdev(PRCAUCs), sep='')
    print('FP: ', sum(fps) ,sep='')
    print('FN: ', sum(fns) ,sep='')