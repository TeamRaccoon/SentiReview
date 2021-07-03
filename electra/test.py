import os, sys, json
sys.path.append('/home/ubuntu/electra')
from predict_line import predict_line

def main():
    with open("conf.json") as f:
        args = json.load(f)

    print(args)

    args['root_dir'] = os.getcwd()

    print(args['root_dir'])

def test():
    str = '노잼'

    pred = predict_line(str)

    print('predict: ',pred)

if __name__ == '__main__':
    test()
