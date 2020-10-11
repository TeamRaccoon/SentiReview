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
    str = '섹스만 있고 다른게없는 연인은 헤어지게 되어 있음 일본 av보다 못한 작품'

    pred = predict_line(str)

    print('predict: ',pred)

if __name__ == '__main__':
    test()
