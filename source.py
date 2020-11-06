"""
tensorflow-gpu 2.2.0
"""
import argparse

from handout import *

parser = argparse.ArgumentParser()

embeddingGroup = parser.add_argument_group('embedding')
embeddingGroup.add_argument('--embeddingPath', type=str, default=None)
embeddingGroup.add_argument('--featureSpecified', type=bool, default=False)
embeddingGroup.add_argument('--features', type=list, default=list(range(10)))

trainGroup = parser.add_argument_group('train')
trainGroup.add_argument('--train', type=bool, default=False)
trainGroup.add_argument('--model', type=str, default='SABiLSTM')
trainGroup.add_argument('--filename', type=str, default='handout/ecoli.csv')
trainGroup.add_argument('--embeddingModelPath', type=str, default='handout/model_300dim.pkl')
trainGroup.add_argument('--optimizer', type=str, default='Adam')
trainGroup.add_argument('--learningRate', type=float, default=0.001)
trainGroup.add_argument('--epochs', type=int, default=1)
trainGroup.add_argument('--batchSize', type=int, default=64)
trainGroup.add_argument('--plot', type=bool, default=False)
trainGroup.add_argument('--rateOfValid', type=float, default=0.5)

evaluateGroup = parser.add_argument_group('evaluate')
evaluateGroup.add_argument('--evaluate', type=bool, default=True)
evaluateGroup.add_argument('--evaluateEmbeddings', type=str, default=None)
evaluateGroup.add_argument('--evaluateEmbeddingsVec', type=str, default=None)
evaluateGroup.add_argument('--evaluateBatchSize', type=int, default=501)
evaluateGroup.add_argument('--evaluateRateOfValid', type=float, default=0.5)
evaluateGroup.add_argument('--evaluateRepeatedTimes', type=int, default=20)

saveGroup = parser.add_argument_group('save')
saveGroup.add_argument('--save', type=bool, default=False)
saveGroup.add_argument('--saveModelPath', type=str, default=None)

resumeGroup = parser.add_argument_group('resume')
resumeGroup.add_argument('--resume', type=bool, default=False)
resumeGroup.add_argument('--resumeModelPath', type=str, default='handout/trainedSABiLSTMModel.pkl')

SABiLSTMGroup = parser.add_argument_group('SABiLSTM')
SABiLSTMGroup.add_argument('--attentionDimension', type=int, default=32)
SABiLSTMGroup.add_argument('--multipleAttentions', type=int, default=40)

gpuGroup = parser.add_argument_group('--gpu')
gpuGroup.add_argument('--memoryGrowth', type=bool, default=False)


parser = parser.parse_args()

if __name__ == '__main__':
    if parser.train is True or parser.resume is True:
        if parser.model == "SABiLSTM":
            if parser.saveModelPath == None:
                parser.saveModelPath = 'handout/trainedSABiLSTMModel.pkl'
            if parser.embeddingPath == None:
                parser.embeddingPath = 'handout/datasetEmbeddingsWithTensor.pkl'
            model = trainSABiLSTMModel(parser)
        elif parser.model == "RF":
            if parser.saveModelPath == None:
                parser.saveModelPath = 'handout/pseudomonas/trainedRFMModel.pkl'
            if parser.embeddingPath == None:
                parser.embeddingPath  = 'handout/pseudomonas/datasetEmbeddingsWithVector.pkl'
            if parser.evaluateEmbeddings == None:
                parser.evaluateEmbeddings = "handout/pseudomonas/datasetEmbeddingsWithVector.pkl"
            model = trainRFModel(parser)
        else:
            raise RuntimeError
    if parser.evaluate is True:
        if parser.model == "RF":
            evaluateRFModel(parser)
        if parser.model == "SABiLSTM":
            evaluateSABiLSTMModel(parser)
        if parser.model =="Ensemble":
            evaluateEnsembleModel(parser)
