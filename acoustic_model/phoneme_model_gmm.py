import os
import joblib
import numpy as np
from hmmlearn.hmm import GMMHMM

train_dir = '../phoneme_data_mfcc/TRAIN'
test_dir = '../phoneme_data_mfcc/TEST'
output_dir = '../phoneme_models_gmm'
os.makedirs(output_dir, exist_ok=True)


for dir in os.listdir(train_dir):
    xtrain_dir = os.path.join(train_dir, dir)
    xtest_dir = os.path.join(test_dir, dir)

    X_train = []
    train_lengths = []
    for file in os.listdir(xtrain_dir):
        filedir = os.path.join(xtrain_dir, file)
        X_train.extend(np.load(filedir))
        train_lengths.append(len(np.load(filedir)))
    
    X_test = []
    test_lengths = []
    for file in os.listdir(xtest_dir):
        filedir = os.path.join(xtest_dir, file)
        X_test.extend(np.load(filedir))
        test_lengths.append(len(np.load(filedir)))

    best_score = best_model = None
    n_fits = 5
    np.random.seed(13)
    for idx in range(n_fits):
        model = GMMHMM(n_components=3, n_mix = 3, init_params='smc')
        model.transmat_ = [[0,1,0],[0,0,1],[0,0,1]]
        model.fit(X_train, train_lengths)
        score = model.score(X_test)
        print(f'Model #{idx}\tScore: {score}')
        if best_score is None or score > best_score:
            best_model = model
            best_score = score
    print(f'Best score:      {best_score}')

    output_file = str(output_dir) + "/" + str(dir)
    joblib.dump(best_model, output_file)

    i = 0
    sum = 0
    for score_dir in os.listdir(test_dir):
        xscore_dir = os.path.join(test_dir, score_dir)
        score_test = []
        score_lengths = []
        for file in os.listdir(xscore_dir):
            filedir = os.path.join(xscore_dir, file)
            score_test.extend(np.load(filedir))
            score_lengths.append(len(np.load(filedir)))
        score = best_model.score(score_test)
        sum += score
        i += 1
    print(sum/i)