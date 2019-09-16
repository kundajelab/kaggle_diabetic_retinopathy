import gzip
import numpy as np
from scipy.special import softmax

#average over the different augmentations
def load_deterministic_labels(pred_folder):
    subfolder_names = [
        pred_folder+"/xyflip-False_horizontalflip-False_verticalflip-False",
        pred_folder+"/xyflip-False_horizontalflip-False_verticalflip-True",
        pred_folder+"/xyflip-False_horizontalflip-True_verticalflip-False",
        pred_folder+"/xyflip-False_horizontalflip-True_verticalflip-True",
        pred_folder+"/xyflip-True_horizontalflip-False_verticalflip-False",
        pred_folder+"/xyflip-True_horizontalflip-False_verticalflip-True",
        pred_folder+"/xyflip-True_horizontalflip-True_verticalflip-False",
        pred_folder+"/xyflip-True_horizontalflip-True_verticalflip-True"
    ]
    softmax_logits = []
    for subfolder in subfolder_names:
        softmax_logits.append(
            np.array([[float(y) for y in x.decode("utf-8").split("\t")[1:]]
                     for x in gzip.open(subfolder+"/deterministic_preds.txt.gz", 'rb')]))
    softmax_logits = np.mean(softmax_logits, axis=0)
    return softmax_logits

kaggle_labels_names = [x.decode("utf-8").split("\t")[0].split("/")[-1]
                       for x in gzip.open("../valid_labels.txt.gz", 'rb')] 
kaggle_labels = np.array([int(x.decode("utf-8").split("\t")[1])
                              for x in gzip.open("../valid_labels.txt.gz", 'rb')])
kaggle_softmax_logits = load_deterministic_labels("../kaggle_preds")
kaggle_softmax_preds = softmax(kaggle_softmax_logits, axis=-1)
from sklearn.metrics import roc_auc_score
kaggle_binary_labels = 1.0*(kaggle_labels > 0.0)
kaggle_binary_preds = 1-kaggle_softmax_preds[:,0]
kaggle_binary_logits = (np.log(np.maximum(kaggle_binary_preds,1e-7))
                        -np.log(np.maximum(1-kaggle_binary_preds, 1e-7)))
print(roc_auc_score(y_true=kaggle_binary_labels,
                    y_score=kaggle_binary_preds))

open("kaggle.txt", 'w').write(
"\n".join([(x+"\t"+str(y)
             +"\t"+(",".join([str(w) for w in z]))) 
              for (x, y, z) in
              zip(kaggle_labels_names,
                  kaggle_labels, kaggle_softmax_logits)]))


messidor_labels_names = [
    x[1].decode("utf-8").split("\t")[0]
    for x in enumerate(gzip.open(
        "../messidor_preds/messidor_labels_withcorrections.txt.gz", 'rb'))
    if x[0] > 0] 
messidor_labels = np.array([
    int(x[1].decode("utf-8").split("\t")[2]) for x in
    enumerate(gzip.open("../messidor_preds/"
                        +"messidor_labels_withcorrections.txt.gz", 'rb'))
    if x[0] > 0])
messidor_softmax_logits = load_deterministic_labels("../messidor_preds")
messidor_softmax_preds = softmax(messidor_softmax_logits, axis=-1)
from sklearn.metrics import roc_auc_score
messidor_binary_labels = 1.0*(messidor_labels > 0.0)
messidor_binary_preds = 1-messidor_softmax_preds[:,0]
messidor_binary_logits = (np.log(np.maximum(messidor_binary_preds,1e-7))
                               -np.log(np.maximum(1-messidor_binary_preds,1e-7)))
print(roc_auc_score(y_true=messidor_binary_labels,
                    y_score=messidor_binary_preds))
open("messidor.txt", 'w').write(
"\n".join([(x+"\t"+str(y)
             +"\t"+(",".join([str(w) for w in z]))) 
              for (x, y, z) in
              zip(messidor_labels_names, messidor_labels,
                  messidor_softmax_logits)]))
