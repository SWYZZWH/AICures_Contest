import os
import dill
import pickle
import matplotlib.pyplot as plt

import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses
from sklearn.metrics import auc, roc_auc_score, roc_curve, precision_recall_curve, accuracy_score
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.linear_model import LogisticRegression
from .data import embedData2Vector, embedData2Tensor, generateBalancedData


def loadData(parser, isTest = False):
    if isTest and parser.evaluateEmbeddings:
       parser.embeddingPath = parser.evaluateEmbeddings

    if os.path.exists(parser.embeddingPath):
        embeddingResults = pickle.load(open(parser.embeddingPath, 'rb'))
    else:
        filename = parser.filename
        embeddingModelPath = parser.embeddingModelPath
        embeddingResults = embedData2Tensor(filename, embeddingModelPath)

    if parser.featureSpecified:
        features = parser.features
        embeddings = [tf.constant(np.vstack([embedding[:, feature] for feature in features]).transpose())
                      for embedding in embeddingResults['embeddings']]
    else:
        embeddings = [tf.constant(embedding) for embedding in embeddingResults['embeddings']]

    smiles = embeddingResults['smiles']
    activities = tf.constant(embeddingResults['activities'])
    print(parser.embeddingPath)
    return smiles, embeddings, activities

def evaluateEnsembleModel(parser):
    #get rf and lr models
    embeddingPath = parser.evaluateEmbeddingsVec
    embeddingResults = pickle.load(open(embeddingPath, "rb"))
    testEmbeddings = embeddingResults["embeddings"]
    testActivities = embeddingResults["activities"]
    model_RF, model_LR = trainRFModel(parser, isEnsemble=True)
    results_RF = model_RF.predict_proba(testEmbeddings)[:, 1]
    results_LR = model_LR.predict_proba(testEmbeddings)[:, 1]

    #get LSTM model
    _, embeddings, activities = loadData(parser, isTest=True)
    if not os.path.exists(parser.saveModelPath):
        print("Please Train SABiLSTM model first using train_lstm func in 10foldtest!")
        exit(1)
    model_lstm = dill.load(open(parser.saveModelPath, 'rb'))
    results_lstm = [model_lstm(tf.expand_dims(embedding, 0)) for embedding in embeddings]
    [results_lstm] = np.vstack([result.numpy() for result in results_lstm]).transpose()

    #ensemble
    results = results_RF
    for i in range(len(testEmbeddings)):
        if results[i] < 0.5 and results_LR[i] > 0.5:
            results[i] = results[i]+0.2
        if results[i] < 0.5 and results_RF[i] > 0.5:
            results[i] = results[i]+0.2
    results = [(results_LR[i]+2*results_RF[i]+results_lstm[i])/4 for i in range(len(testEmbeddings))]

    #evalutate
    ACC = 0
    fp = 0
    fn = 0
    for i in range(len(results)):
        if results[i]> 0.5 and testActivities[i] == 0:
            fp += 1
        elif results[i] <= 0.5 and testActivities[i] == 1:
            fn += 1
    print("fp:", fp)
    print("fn:", fn)
    fpr, tpr, _ = roc_curve(testActivities, results)
    ROCAUC = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(testActivities, results)
    PRCAUC = auc(recall, precision)
    if parser.plot:
        plt.figure()
        plt.plot(fpr, tpr, marker='o')
        plt.title('ROC of ' + parser.model + ' model')
        plt.savefig('handout/pseudomonas/RF-ROC-AUC.png')
        plt.figure()
        plt.plot(recall, precision, marker='o')
        plt.title('PRC of ' + parser.model + ' model')
        plt.savefig('handout/pseudomonas/RF-PRC-AUC.png')
        plt.show()
    print('ROC-AUC: ', ROCAUC)
    print('PRC-AUC: ', PRCAUC)

    # save statistics to file
    with open("handout/pseudomonas/train_cv/statistics.txt", "a") as f:
        f.write(" ".join(map(str, [ACC, ROCAUC, PRCAUC, fp, fn]))+"\n")

    return

def trainRFModel(parser, isEnsemble = False):
    embeddingPath = parser.embeddingPath
    if os.path.exists(embeddingPath):
        try:
            embeddingResults = pickle.load(open(embeddingPath, 'rb'))
        except:
            raise RuntimeError("Can't load embedding vectors!")
    else:
        raise RuntimeError("Can't find embedding file!")

    embeddingResults = generateBalancedData(embeddingResults, parser.rateOfValid)
    trainEmbeddings = embeddingResults['embeddings']
    trainActivities = embeddingResults["activities"]
    model_RF = RF(n_estimators=100, random_state=0,oob_score=True)
    model_LR = LogisticRegression(max_iter = 10000)
    model_RF.fit(trainEmbeddings, trainActivities)
    model_LR.fit(trainEmbeddings, trainActivities)
    model = model_LR
    print("Training is over!")

    if isEnsemble:
        return model_RF, model_LR

    if parser.save:
        pickle.dump(model, open(parser.saveModelPath, 'wb'))
    else:
        saveOrNot = input('Save of not (y/n): ')
        if saveOrNot is 'y' or saveOrNot is 'Y':
            dill.dump(model, open(parser.saveModelPath, 'wb'))
        elif saveOrNot is not 'n' and saveOrNot is not 'N':
            raise RuntimeError

    return

def evaluateRFModel(parser):
    embeddingPath = parser.evaluateEmbeddings
    if os.path.exists(embeddingPath):
        try:
            embeddingResults = pickle.load(open(embeddingPath, "rb"))
        except:
            raise RuntimeError("Can't load embedding vectors!")
    else:
        raise RuntimeError("Can't find embedding file!")
    modelPath = parser.saveModelPath
    if os.path.exists(embeddingPath):
        try:
            model = pickle.load(open(modelPath, "rb"))
        except:
            raise RuntimeError("Can't load model!")
    else:
        raise RuntimeError("Can't find model file!")

    testEmbeddings = embeddingResults["embeddings"]
    testActivities = embeddingResults["activities"]
    ACC = model.score(testEmbeddings, testActivities)
    print("accuracy:", ACC)

    fp = 0
    fn = 0
    results = model.predict_proba(testEmbeddings)[:, 1]
    predicts = model.predict(testEmbeddings)
    for i in range(len(predicts)):
        if predicts[i] != testActivities[i]:
            if predicts[i] == 1:
                fp += 1
            else:
                fn += 1
    print("fp:", fp)
    print("fn:", fn)
    try:
        fpr, tpr, _ = roc_curve(testActivities, results)
        ROCAUC = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(testActivities, results)
        PRCAUC = auc(recall, precision)
        if parser.plot:
            plt.figure()
            plt.plot(fpr, tpr, marker='o')
            plt.title('ROC of ' + parser.model + ' model')
            plt.savefig('handout/pseudomonas/RF-ROC-AUC.png')
            plt.figure()
            plt.plot(recall, precision, marker='o')
            plt.title('PRC of ' + parser.model + ' model')
            plt.savefig('handout/pseudomonas/RF-PRC-AUC.png')
            plt.show()
    except ValueError:
        print("error!")
        ROCAUC = 0.5
        PRCAUC = 0

    print('ROC-AUC: ', ROCAUC)
    print('PRC-AUC: ', PRCAUC)

    # save statistics to file
    with open("handout/pseudomonas/train_cv/statistics.txt", "a") as f:
        f.write(" ".join(map(str, [ACC, ROCAUC, PRCAUC, fp, fn]))+"\n")

    return



class SABiLSTMModel(keras.Model):
    """
    Self-Attention Bidirectional Long Short-Term Memory model.
    """
    def __init__(self, parser):
        super().__init__()
        self.lstmUnit = len(parser.features) if parser.featureSpecified else 300
        self.attentionDimension = parser.attentionDimension
        self.multipleAttentions = parser.multipleAttentions
        self.biLSTM = layers.Bidirectional(layers.LSTM(self.lstmUnit, return_sequences=True, activation=tf.nn.sigmoid))
        self.selfAttention = SABiLSTMModel.selfAttentionLayer(self.lstmUnit, self.attentionDimension,
                                                              self.multipleAttentions)
        self.dense = layers.Dense(1)

    @tf.function(experimental_relax_shapes=True)
    def call(self, embedding):
        H = self.biLSTM(embedding)
        H = tf.squeeze(H, axis=0)
        A = self.selfAttention(H)
        Ma = tf.matmul(A, H)
        output = tf.sigmoid(tf.reduce_mean(self.dense(Ma)))
        return output

    class selfAttentionLayer(keras.layers.Layer):
        def __init__(self, lstmUnit=None, attentionDimension=None, multipleAttentions=None):
            if lstmUnit is None or attentionDimension is None or multipleAttentions is None:
                raise RuntimeError('Attention dimension and multiple attentions must be specified')
            super().__init__()
            self.lstmUnit = lstmUnit
            self.attentionDimension = attentionDimension
            self.multipleAttentions = multipleAttentions
            self.W1 = tf.Variable(tf.nn.softmax(tf.random.uniform([self.attentionDimension, 2 * self.lstmUnit])))
            self.W2 = tf.Variable(tf.random.uniform([self.multipleAttentions, self.attentionDimension]))

        @tf.function
        def call(self, H):
            A = tf.nn.softmax(tf.matmul(self.W2, tf.nn.tanh(tf.matmul(self.W1, tf.transpose(H)))))
            return A


def trainSABiLSTMModel(parser):
    """
    Train the Self-Attention Bidirectional Long Short-Term Memory model.
    """

    @tf.function(experimental_relax_shapes=True)
    def trainBatch(model, optimizer, batchEmbeddings, batchActivities):
        with tf.GradientTape() as tape:
            results = [model(tf.expand_dims(embedding, 0)) for embedding in batchEmbeddings]
            loss = tf.reduce_mean(losses.binary_crossentropy(batchActivities, results))

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss, results

    if parser.memoryGrowth:
        gpu = tf.config.experimental.get_visible_devices('GPU')[0]
        tf.config.experimental.set_memory_growth(device=gpu, enable=True)

    if os.path.exists(parser.embeddingPath):
        embeddingResults = pickle.load(open(parser.embeddingPath, 'rb'))
    else:
        filename = parser.filename
        embeddingModelPath = parser.embeddingModelPath
        embeddingResults = embedData2Tensor(filename, embeddingModelPath)

    embeddingResults = generateBalancedData(embeddingResults, parser.rateOfValid)

    if parser.featureSpecified:
        features = parser.features
        embeddings = [tf.constant(np.vstack([embedding[:, feature] for feature in features]).transpose())
                      for embedding in embeddingResults['embeddings']]
    else:
        embeddings = [tf.constant(embedding) for embedding in embeddingResults['embeddings']]

    activities = tf.constant(embeddingResults['activities'])
    n = len(embeddings)

    if parser.resume and os.path.exists(parser.resumeModelPath):
        model = dill.load(open(parser.resumeModelPath, 'rb'))
    else:
        model = SABiLSTMModel(parser)

    epochs = parser.epochs
    batchSize = parser.batchSize

    if parser.optimizer == 'Adam':
        optimizer = optimizers.Adam(parser.learningRate)
    elif parser.optimizer == 'SGD':
        optimizer = optimizers.SGD(parser.learningRate)
    elif parser.optimizer == 'RMSprop':
        optimizer = optimizers.RMSprop(parser.learningRate)
    else:
        raise RuntimeError

    trainingLosses = []
    accuracies = []
    if parser.plot:
        plt.figure()
        for epoch in range(epochs):
            for step in range(int(np.floor(n / batchSize))):
                beg = step * batchSize
                end = (step + 1) * batchSize if (step + 1) * batchSize < n else n
                batchEmbeddings = embeddings[beg: end]
                batchActivities = activities[beg: end]
                loss, results = trainBatch(model, optimizer, batchEmbeddings, batchActivities)
                [results] = np.vstack([result.numpy() for result in results]).transpose()
                prediction = np.where(np.array(results) < 0.5, tf.zeros_like(results), tf.ones_like(results))
                accuracy = np.mean(batchActivities.numpy() == prediction)
                trainingLosses.append(loss)
                accuracies.append(accuracy)
                print(accuracy)
                print(loss.numpy())
                plt.cla()
                plt.plot()
                accuracies.append(accuracy)
    else:
        for epoch in range(epochs):
            for step in range(int(np.floor(n / batchSize))):
                beg = step * batchSize
                end = (step + 1) * batchSize if (step + 1) * batchSize < n else n
                batchEmbeddings = embeddings[beg: end]
                batchActivities = activities[beg: end]
                loss, results = trainBatch(model, optimizer, batchEmbeddings, batchActivities)
                loss = loss.numpy()
                [results] = np.vstack([result.numpy() for result in results]).transpose()
                prediction = np.where(np.array(results) < 0.5, tf.zeros_like(results), tf.ones_like(results))
                accuracy = np.mean(batchActivities.numpy() == prediction)
                trainingLosses.append(loss)
                accuracies.append(accuracy)
                print('Epoch:', epoch + 1, 'step:', step + 1)
                print('Batch loss:', loss, '\nBatch accuracy:', accuracy)
                print('Batch prediction:', [int(pred) for pred in prediction])
                print('Batch activities:', batchActivities.numpy().tolist())
                print('logits:', results, end='\n\n')

    if parser.save:
        dill.dump(model, open(parser.saveModelPath, 'wb'))
    else:
        saveOrNot = input('Save of not (y/n): ')
        if saveOrNot is 'y' or saveOrNot is 'Y':
            dill.dump(model, open(parser.saveModelPath, 'wb'))
        elif saveOrNot is not 'n' and saveOrNot is not 'N':
            raise RuntimeError

    return model

def evaluateSABiLSTMModel(parser):
    if parser.memoryGrowth:
        gpu = tf.config.experimental.get_visible_devices('GPU')[0]
        tf.config.experimental.set_memory_growth(device=gpu, enable=True)

    _, embeddings, activities = loadData(parser,isTest=True)
    model = dill.load(open(parser.saveModelPath, 'rb'))
    N = len(embeddings)
    allIndices = np.arange(N)
    np.random.shuffle(allIndices)
    indices = allIndices[0:parser.evaluateBatchSize]
    batchEmbeddings = [embeddings[index] for index in indices]
    batchActivities = [activities[index] for index in indices]
    results = [model(tf.expand_dims(embedding, 0)) for embedding in batchEmbeddings]
    [results] = np.vstack([result.numpy() for result in results]).transpose()

    fp, fn = 0, 0
    try:
        prediction = np.where(np.array(results) < 0.5, tf.zeros_like(results), tf.ones_like(results))
        for i in range(len(prediction)):
            if prediction[i] != batchActivities[i]:
                if prediction[i] == 1:
                    fp += 1
                else:
                    fn += 1
        print("fp:", fp)
        print("fn:", fn)
        ACC = np.mean(batchActivities == prediction)
        ROCAUC = roc_auc_score(np.array(batchActivities), results)
        precision, recall, _ = precision_recall_curve(np.array(batchActivities), results)
        PRCAUC = auc(recall, precision)
    except:
        ROCAUC = 0.5
        PRCAUC = 0

    print("ACC:", ACC)
    print('ROC-AUC: ', ROCAUC)
    print('PRC-AUC: ', PRCAUC)

    # save statistics to file
    with open("handout/pseudomonas/train_cv/statistics.txt", "a") as f:
        f.write(" ".join(map(str, [ACC, ROCAUC, PRCAUC, fp, fn])) + "\n")

    return