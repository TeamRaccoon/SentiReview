import os, sys, json
import numpy as np

sys.path.append('/home/ubuntu/electra')
from sentireview import vote

def predict_line(str):
    # convert 1 line to list with 1 element
    line = [[0, str, 1]]
    print(line)

    # get config from config file
    conf = load_from_conf()

    # voting from designated model
    predict, _ = vote(conf, line)

    prdict = predict.tolist()

    return predict

def load_from_conf():

    with open("conf.json") as f:
        conf = json.load(f)

    conf["root_dir"] = '/home/ubuntu/electra'
    conf["test_model_path"] = os.path.join(conf["root_dir"],"output","CNN-biLSTM")

    return conf
