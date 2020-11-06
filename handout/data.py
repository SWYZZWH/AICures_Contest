import csv
import pickle
import numpy as np
from rdkit.Chem import MolFromSmiles
from gensim.models import word2vec
from mol2vec.features import MolSentence, mol2alt_sentence, sentences2vec


def embedData2Vector(filename=None, modelPath=None, outputPath=None,  r=None, uncommon=None):
    """
    embedData2Vector: Embed the dataset to vectors using the pretrained mol2vec model.
    """
    if filename is None:
        filename = 'handout/ecoli.csv'
    if modelPath is None:
        modelPath = 'handout/model_300dim.pkl'
    if outputPath is None:
        outputPath = 'handout/datasetEmbeddingsWithVec.pkl'
    if r is None:
        r = 1

    smiles, activities = [], []
    with open(filename) as file:
        reader = csv.reader(file)
        for row in reader:
            #需要指定列参数
            smiles.append(row[0])
            activities.append(row[1])

    model = word2vec.Word2Vec.load(modelPath)
    mols = [MolFromSmiles(smile) for smile in smiles[1:]]
    sentences = [MolSentence(mol2alt_sentence(mol, r)) for mol in mols]
    vectors = sentences2vec(sentences, model, unseen=uncommon)

    indices = np.arange(len(vectors))
    #np.random.shuffle(indices)
    smiles = [smiles[idx + 1] for idx in indices]
    vectors = np.array([vectors[idx] for idx in indices])
    activities = [int(activities[idx + 1]) for idx in indices]
    results = {'smiles': smiles, 'embeddings': vectors, 'activities': activities}
    pickle.dump(results, open(outputPath, 'wb'))
    return results


def embedData2Tensor(filename=None, modelPath=None, outputPath=None, r=None, uncommon=None):
    """
    embedDataset2Tensor: Embed the dataset to tensors using the pretrained mol2vec model.
    """
    if filename is None:
        filename = 'handout/ecoli.csv'
    if modelPath is None:
        modelPath = 'handout/model_300dim.pkl'
    if outputPath is None:
        outputPath = 'handout/datasetEmbeddingsWithTensor.pkl'
    if r is None:
        r = 1

    smiles, activities = [], []
    with open(filename) as file:
        reader = csv.reader(file)
        for row in reader:
            smiles.append(row[0])
            activities.append(row[1])

    model = word2vec.Word2Vec.load(modelPath)
    mols = [MolFromSmiles(smile) for smile in smiles[1:]]
    sentences = [MolSentence(mol2alt_sentence(mol, r)) for mol in mols]

    keys = set(model.wv.vocab.keys())
    tensors = []
    if uncommon:
        unseen_vec = model.wv.word_vec(uncommon)
        for sentence in sentences:
            tensors.append(np.vstack([model.wv.word_vec(ch) if ch in set(sentence) & keys
                                      else unseen_vec for ch in sentence]))
    else:
        for sentence in sentences:
            tensors.append(np.vstack([model.wv.word_vec(ch) for ch in sentence
                                      if ch in set(sentence) & keys]))

    indices = np.arange(len(tensors))
    #np.random.shuffle(indices)
    smiles = [smiles[idx + 1] for idx in indices]
    tensors = [tensors[idx] for idx in indices]
    activities = [int(activities[idx + 1]) for idx in indices]
    results = {'smiles': smiles, 'embeddings': tensors, 'activities': activities}
    pickle.dump(results, open(outputPath, 'wb'))
    return results

def genVecFor10Folds(modelPath=None):
    """
    generate embedding vectors for csv in  fold_0 - fold_10
    """
    print("Generate data...")
    for i in range(10):
        folder = "handout/pseudomonas/train_cv/fold_" + str(i)
        dev_file, train_file, test_file = folder + "/dev.csv", folder + "/train.csv", folder + "/test.csv"
        dev_output, train_output, test_output = folder + "/datasetEmbeddingsWithVecPseDev.pkl", \
                                                folder + "/datasetEmbeddingsWithVecPseTrain.pkl", folder + "/datasetEmbeddingsWithVecPseTest.pkl"
        embedData2Vector(dev_file,modelPath,dev_output)
        embedData2Vector(train_file,modelPath,train_output)
        embedData2Vector(test_file,modelPath,test_output)
    return

def genTenorFor10Folds(modelPath=None):
    """
    generate embedding tensors for csv in  fold_0 - fold_10
    """
    print("Generate data...")
    for i in range(10):
        folder = "handout/pseudomonas/train_cv/fold_" + str(i)
        dev_file, train_file, test_file = folder + "/dev.csv", folder + "/train.csv", folder + "/test.csv"
        dev_output, train_output, test_output = folder + "/datasetEmbeddingsWithTensorPseDev.pkl", \
                                                folder + "/datasetEmbeddingsWithTensorPseTrain.pkl", folder + "/datasetEmbeddingsWithTensorPseTest.pkl"
        embedData2Tensor(dev_file,modelPath,dev_output)
        embedData2Tensor(train_file,modelPath,train_output)
        embedData2Tensor(test_file,modelPath,test_output)
    return

def generateBalancedData(embeddingResults, rateOfValid):
    """
    generateBalancedData: generate balanced data using the embedding results and rate of valid data.
    """
    smiles = embeddingResults['smiles']
    embeddings = embeddingResults['embeddings']
    activities = embeddingResults['activities']

    indicesOfValid = np.argwhere(activities == np.ones_like(activities)).tolist()
    indicesOfInvalid = np.argwhere(activities == np.zeros_like(activities)).tolist()
    validCount = int(np.floor(len(indicesOfInvalid) * rateOfValid / (1 - rateOfValid)))
    choosenValid = np.random.randint(0, len(indicesOfValid), validCount).tolist()

    newSmiles = [smiles[indicesOfValid[choosen][0]] for choosen in choosenValid]
    newSmiles.extend([smiles[index[0]] for index in indicesOfInvalid])
    newEmbeddings = [embeddings[indicesOfValid[choosen][0]] for choosen in choosenValid]
    newEmbeddings.extend([embeddings[index[0]] for index in indicesOfInvalid])
    newActivities = [activities[indicesOfValid[choosen][0]] for choosen in choosenValid]
    newActivities.extend([activities[index[0]] for index in indicesOfInvalid])

    indices = list(range(len(newEmbeddings)))
    np.random.shuffle(indices)
    newSmiles = [newSmiles[index] for index in indices]
    newEmbeddings = [newEmbeddings[index] for index in indices]
    newActivities = [newActivities[index] for index in indices]
    newEmbeddingResults = {'smiles': newSmiles, 'embeddings': newEmbeddings, 'activities': newActivities}

    return newEmbeddingResults
