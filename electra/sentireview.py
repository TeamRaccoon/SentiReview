import os, sys, json
import random
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings(action='ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

sys.path.append('/home/ubuntu/electra/transformers')
from activations import get_activation
from configuration_electra import ElectraConfig
from tokenization_electra import ElectraTokenizer
from modeling_electra import ElectraPreTrainedModel, ElectraModel
from optimization import AdamW, get_linear_schedule_with_warmup

sys.path.append('/home/ubuntu/electra/fine_tuning')
from util import clean_text
from hanspell import spell_checker
import re

#electra: base 기준 12 layers
class ElectraForSequenceClassification(ElectraPreTrainedModel):
    def __init__(self, config): #이 모델에서 사용할 layer 선언
        super().__init__(config)

        # 분류할 라벨의 개수 #parameter 정의
        self.num_labels = config.num_labels

        # ELECTRA 모델
        self.electra = ElectraModel(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.linear_1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_2 = nn.Linear(config.hidden_size, config.num_labels)

        self.softmax = nn.Softmax(dim=-1)
        #layer 선언
        
        num_filter=16
        kernel_size_ls = [3,4,5]
        self.CNN = nn.ModuleList([nn.Conv2d(1, num_filter, (kernel_size, config.hidden_size)) for kernel_size in kernel_size_ls])
        self.linear_cnn = nn.Sequential(
            nn.Linear(len(kernel_size_ls)*num_filter, config.hidden_size), 
            nn.ReLU(), 
            nn.Dropout(config.hidden_dropout_prob)
        )

        self.biLSTM = nn.LSTM(input_size=config.hidden_size, hidden_size=int(config.hidden_size/2), num_layers=1, dropout=config.hidden_dropout_prob, batch_first=True, bidirectional=True)
        
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        discriminator_hidden_states = self.electra(input_ids, attention_mask, token_type_ids)

        #print(discriminator_hidden_states.shape) 

        # (batch_size, max_length, hidden_size)
        discriminator_hidden_states = discriminator_hidden_states[0] #맨 위층 layer     

        # (batch_size, max_length, hidden_size) -> (batch_size, hidden_size)
        lstm_output, (hidden, cell) = self.biLSTM(discriminator_hidden_states)
        #cls_output = lstm_output[:, 0, :] #[batch, length, hidden]
        cls_output = self.dropout(lstm_output)
  
        cls_output.unsqueeze_(1)
        conved = [conv(cls_output).squeeze(3) for conv in self.CNN]
        pooled = [F.max_pool1d(conv, (conv.size(2))).squeeze(2) for conv in conved]
        concated = torch.cat(pooled, dim = 1)
        cls_output = self.linear_cnn(concated)

        # (batch_size, hidden_size) -> (batch_size, hidden_size)
        cls_output = self.linear_1(cls_output)
        cls_output = get_activation("gelu")(cls_output)
        cls_output = self.dropout(cls_output)

        # (batch_size, hidden_size) -> (batch_size, num_labels)
        cls_output = self.linear_2(cls_output)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(cls_output, labels) #*****loss_fct(예측값: 2차원(batch, 2:0,1), 정답값: 1차원(batch))*****

            return loss, self.softmax(cls_output)
        else:
            return self.softmax(cls_output)


# 학습 or 평가 데이터를 읽어 리스트에 저장
def read_data(file_path):
    with open(file_path, "r", encoding="utf8") as inFile:
        lines = inFile.readlines()

    datas = []
    for index, line in enumerate(tqdm(lines, desc="read_data")):
        # 입력 문장을 \t으로 분리
        pieces = line.strip().split("\t")

        # 데이터의 형태가 올바른지 체크
        #assert len(pieces) == 3
        if len(pieces) != 3:
            continue

        if(index == 0):
            continue

        pieces[1] = clean_text(os.path.join(config["root_dir"],"data"), pieces[1])

        id, sequence, label = pieces[0], pieces[1], int(pieces[2])
        datas.append((id, sequence, label))

    return datas


def convert_data2dataset(datas, tokenizer, max_length):
    total_input_ids, total_attention_mask, total_token_type_ids, total_labels = [], [], [], []
    for index, data in enumerate(tqdm(datas, desc="convert_data2dataset")):
        _, sequence, label = data
        tokens = tokenizer.tokenize(sequence)

        tokens = ["[CLS]"] + tokens
        tokens = tokens[:max_length-1]
        tokens.append("[SEP]")

        input_ids = [tokenizer._convert_token_to_id(token) for token in tokens]
        assert len(input_ids) <= max_length

        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)

        padding = [0] * (max_length - len(input_ids))

        input_ids += padding
        attention_mask += padding
        token_type_ids += padding

        total_input_ids.append(input_ids)
        total_attention_mask.append(attention_mask)
        total_token_type_ids.append(token_type_ids)
        total_labels.append(label)

  #      if(index < 2):
  #          print("*** Example ***")
  #          print("sequence : {}".format(sequence))
  #          print("tokens: {}".format(" ".join([str(x) for x in tokens])))
  #          print("input_ids: {}".format(" ".join([str(x) for x in total_input_ids[-1]])))
  #          print("attention_mask: {}".format(" ".join([str(x) for x in total_attention_mask[-1]])))
  #          print("token_type_ids: {}".format(" ".join([str(x) for x in total_token_type_ids[-1]])))
  #          print("label: {}".format(total_labels[-1]))
  #          print()

    total_input_ids = torch.tensor(total_input_ids, dtype=torch.long)
    total_attention_mask = torch.tensor(total_attention_mask, dtype=torch.long)
    total_token_type_ids = torch.tensor(total_token_type_ids, dtype=torch.long)
    total_labels = torch.tensor(total_labels, dtype=torch.long)

    dataset = TensorDataset(total_input_ids, total_attention_mask, total_token_type_ids, total_labels)

    return dataset

def check_hanspell(dataset):
    fails, checked_data = [], []

    for id, sequence, label in dataset:
        sequence = re.sub('[-&]', '', sequence)
        try:
             checked = spell_checker.check(sequence)
             checked_sequence = checked.checked
             checked_data.append((id, checked_sequence,label))
        except Exception:
             fails.append(id)
             checked_data.append((id, sequence, label))

    re_checked = []
    for id, sequence, label in checked_data:
        if id in fails:
           try:
               checked = spell_checker.check(sequence)
               checked_sequence = checked.checked
               re_checked.append((id, checked_sequence, label))
           except:
               print(id)
               re_checked.append((id, sequence, label))
        else:
           re_checked.append((id, sequence, label))

    checked_tata = re_checked

    return checked_data

def do_train(config, electra_model, optimizer, scheduler, train_dataloader, epoch, global_step):

    # batch 단위 별 loss를 담을 리스트
    losses = []
    # 모델의 출력 결과와 실제 정답값을 담을 리스트
    total_predicts, total_corrects = [], []
    for step, batch in enumerate(tqdm(train_dataloader, desc="do_train(epoch_{})".format(epoch))):

        # batch = tuple(t.cuda() for t in batch)
        batch = tuple(t for t in batch)
        input_ids, attention_mask, token_type_ids, labels = batch[0], batch[1], batch[2], batch[3]
        #input_ids, attention_mask, token_type_ids, labels, feature = batch[0], batch[1], batch[2], batch[3], batch[4]

        # 입력 데이터에 대한 출력과 loss 생성
        loss, predicts = electra_model(input_ids, attention_mask, token_type_ids, labels)

        predicts = predicts.argmax(dim=-1)
        predicts = predicts.cpu().detach().numpy().tolist()
        labels = labels.cpu().detach().numpy().tolist()

        total_predicts += predicts
        total_corrects += labels


        if config["gradient_accumulation_steps"] > 1:
            loss = loss / config["gradient_accumulation_steps"]

        # loss 값으로부터 모델 내부 각 매개변수에 대하여 gradient 계산
        loss.backward()
        losses.append(loss.data.item())

        if (step + 1) % config["gradient_accumulation_steps"] == 0 or \
                (len(train_dataloader) <= config["gradient_accumulation_steps"] and (step + 1) == len(train_dataloader)):

            torch.nn.utils.clip_grad_norm_(electra_model.parameters(), config["max_grad_norm"])

            # 모델 내부 각 매개변수 가중치 갱신
            optimizer.step()
            scheduler.step()

            # 변화도를 0으로 변경
            electra_model.zero_grad()
            global_step += 1

    # 정확도 계산
    accuracy = accuracy_score(total_corrects, total_predicts)

    return accuracy, np.mean(losses), global_step


def do_evaluate(electra_model, test_dataloader, mode):
    # 모델의 입력, 출력, 실제 정답값을 담을 리스트
    total_input_ids, total_predicts, total_corrects = [], [], []
    for step, batch in enumerate(tqdm(test_dataloader, desc="do_evaluate")):

        # batch = tuple(t.cuda() for t in batch)
        batch = tuple(t for t in batch)
        input_ids, attention_mask, token_type_ids, labels = batch[0], batch[1], batch[2], batch[3]

        # 입력 데이터에 대한 출력 결과 생성
        predicts = electra_model(input_ids, attention_mask, token_type_ids)

        predicts = predicts.argmax(dim=-1)
        predicts = predicts.cpu().detach().numpy().tolist()
        labels = labels.cpu().detach().numpy().tolist()
        input_ids = input_ids.cpu().detach().numpy().tolist()

        total_predicts += predicts
        total_corrects += labels
        total_input_ids += input_ids

    # 정확도 계산
    accuracy = accuracy_score(total_corrects, total_predicts)

    if(mode == "train"):
        return accuracy
    else:
        return accuracy, total_input_ids, total_predicts, total_corrects


def train(config):
    # electra config 객체 생성
    electra_config = ElectraConfig.from_pretrained(config["train_model_path"], num_labels=config["num_labels"], cache_dir=config["cache_dir_path"])
    
    # electra tokenizer 객체 생성
    electra_tokenizer = ElectraTokenizer.from_pretrained(config["train_model_path"], do_lower_case=False, cache_dir=config["cache_dir_path"])

    # electra model 객체 생성
    electra_model = ElectraForSequenceClassification.from_pretrained(config["train_model_path"], config=electra_config, cache_dir=config["cache_dir_path"])

    # electra_model.cuda()

    # 학습 데이터 읽기
    train_datas = read_data(file_path=config["train_data_path"])

    # 학습 데이터 전처리
    train_dataset = convert_data2dataset(datas=train_datas, tokenizer=electra_tokenizer, max_length=config["max_length"])

    # 학습 데이터를 batch 단위로 추출하기 위한 DataLoader 객체 생성
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=config["batch_size"])

    # 평가 데이터 읽기
    test_datas = read_data(file_path=config["test_data_path"])

    # 평가 데이터 전처리
    test_dataset = convert_data2dataset(datas=test_datas, tokenizer=electra_tokenizer, max_length=config["max_length"])

    # 평가 데이터를 batch 단위로 추출하기 위한 DataLoader 객체 생성
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=100)

    # 전체 학습 횟수(batch 단위)
    t_total = len(train_dataloader) // config["gradient_accumulation_steps"] * config["epoch"]

    # 모델 학습을 위한 optimizer
    optimizer = AdamW(electra_model.parameters(), lr=config["learning_rate"])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config["warmup_steps"], num_training_steps=t_total)

    if os.path.isfile(os.path.join(config["model_dir_path"], "optimizer.pt")) and os.path.isfile(
            os.path.join(config["model_dir_path"], "scheduler.pt")):
        # 기존에 학습했던 optimizer와 scheduler의 정보 불러옴
        optimizer.load_state_dict(torch.load(os.path.join(config["model_dir_path"], "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(config["model_dir_path"], "scheduler.pt")))

    global_step = 0
    electra_model.zero_grad()
    max_test_accuracy = 0
    for epoch in range(config["epoch"]):
        electra_model.train()

        # 학습 데이터에 대한 정확도와 평균 loss
        train_accuracy, average_loss, global_step = do_train(config=config, electra_model=electra_model,
                                                             optimizer=optimizer, scheduler= scheduler,
                                                             train_dataloader=train_dataloader,
                                                             epoch=epoch+1, global_step=global_step)

        print("train_accuracy : {}\taverage_loss : {}\n".format(round(train_accuracy, 4), round(average_loss, 4)))

        electra_model.eval()

        # 평가 데이터에 대한 정확도
        test_accuracy = do_evaluate(electra_model=electra_model, test_dataloader=test_dataloader, mode=config["mode"])

        print("test_accuracy : {}\n".format(round(test_accuracy, 4)))

        # 현재의 정확도가 기존 정확도보다 높은 경우 모델 파일 저장
        if(max_test_accuracy < test_accuracy):
            max_test_accuracy = test_accuracy

            output_dir = os.path.join(config["model_dir_path"], "checkpoint-{}".format(global_step))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            print("save model in checkpoint-{}\n".format(global_step))
            
            electra_config.save_pretrained(output_dir)
            electra_tokenizer.save_pretrained(output_dir)
            electra_model.save_pretrained(output_dir)
            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

def show_result(total_input_ids, total_predicts, total_corrects, tokenizer, save_file = False):
    if save_file:
        file = open('/gdrive/My Drive/electra/wrong_list.txt',"w+")

    for index, input_ids in enumerate(total_input_ids):
        tokens = [tokenizer._convert_id_to_token(input_id) for input_id in input_ids]

        # [CLS] 토큰 제거
        tokens = tokens[1:]

        # [SEP] 토큰 제거
        tokens = tokens[:tokens.index("[SEP]")]

        # 입력 sequence 복원
        sequence = tokenizer.convert_tokens_to_string(tokens)

        predict, correct = total_predicts[index], total_corrects[index]

        if(predict == 0):
            predict = "negative"
        else:
            predict = "positive"

        if (correct == 0):
            correct = "negative"
        else:
            correct = "positive"

        if save_file:
            file.write("sequence : {}\tpredict : {}\tcorrect : {}\n".format(sequence, predict, correct))

        else:
            print("sequence : {}".format(sequence))
            print("predict : {}".format(predict))
            print("correct : {}".format(correct))
            print()

    if save_file:
        file.close()

def test(config):
    # electra config 객체 생성
    electra_config = ElectraConfig.from_pretrained(os.path.join(config["model_dir_path"], "checkpoint-{}".format(config["checkpoint"])),
                                                   num_labels=config["num_labels"],
                                                   cache_dir=None)
    
    # electra tokenizer 객체 생성
    electra_tokenizer = ElectraTokenizer.from_pretrained(os.path.join(config["model_dir_path"], "checkpoint-{}".format(config["checkpoint"])),
                                                         do_lower_case=False,
                                                         cache_dir=None)
    
    # electra model 객체 생성
    electra_model = ElectraForSequenceClassification.from_pretrained(os.path.join(config["model_dir_path"], "checkpoint-{}".format(config["checkpoint"])),
                                                                     config=electra_config,
                                                                     cache_dir=None)
    print("\nevaluate from checkpoint-{}\n".format(config["checkpoint"]))

    # electra_model.cuda()

    # 평가 데이터 읽기
    test_datas = read_data(file_path=config["test_data_path"])
    
    # 평가 데이터 전처리
    test_dataset = convert_data2dataset(datas=test_datas, tokenizer=electra_tokenizer, max_length=config["max_length"])

    # 평가 데이터를 batch 단위로 추출하기 위한 DataLoader 객체 생성
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=100)

    electra_model.eval()

    # 평가 데이터에 대한 정확도와 모델의 입력, 출력, 정답
    test_accuracy, total_input_ids, total_predicts, total_corrects = do_evaluate(electra_model=electra_model,
                                                                            test_dataloader=test_dataloader,
                                                                            mode=config["mode"])

    print("test_accuracy : {}\n".format(round(test_accuracy, 4)))

    # 10개의 평가 케이스에 대하여 모델 출력과 정답 비교
#    show_result(total_input_ids=total_input_ids[:10], total_predicts=total_predicts[:10],
#                total_corrects=total_corrects[:10], tokenizer=electra_tokenizer)
    w_id, w_pred, w_corr = [], [], []
    for seq, pred, lab in zip(total_input_ids, total_predicts, total_corrects):
        if pred != lab:
            w_id.append(seq)
            w_pred.append(pred)
            w_corr.append(lab)

    show_result(w_id, w_pred, w_corr, electra_tokenizer, config["save_file"])

def eval_from_model(config, model_name, dataset):
    # model name에 따라 electra 객체 생성 > 정확도와 label(정답)리스트 리턴
    if (isinstance(model_name, dict)):
        if (model_name.key() == "nsmc")
            dataset = check_hanspell(dataset)
        _, model_name = model_name

    path = os.path.join(config["test_model_path"],model_name)

    # electra config 객체 생성
    electra_config = ElectraConfig.from_pretrained(path, num_labels=config["num_labels"], cache_dir=None)

    # electra tokenizer 객체 생성
    electra_tokenizer = ElectraTokenizer.from_pretrained(path, do_lower_case=False, cache_dir=None)

    # electra model 객체 생성
    electra_model = ElectraForSequenceClassification.from_pretrained(path, config=electra_config, cache_dir=None)

    print("\n\nevaluate from model: {}".format(model_name))

    # cuda
    # electra_model.cuda()

    # 평가 데이터 전처리
    test_dataset = convert_data2dataset(dataset, tokenizer=electra_tokenizer, max_length=config["max_length"])

    # 평가 데이터를 batch 단위로 추출하기 위한 DataLoader 객체 생성
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=100)

    electra_model.eval()

    # 평가 데이터에 대한 정확도와 모델의 입력, 출력, 정답
    test_accuracy, total_input_ids, total_predicts, total_corrects = do_evaluate(electra_model=electra_model,
                                                                            test_dataloader=test_dataloader,
                                                                            mode=config["mode"])
    return test_accuracy, total_predicts


def vote(config,dataset):
    predict_labels = []
    model_names = config['model_names']

    # names = dict(model_type, model_name(path_name))
    for names in model_names.items():
        acc, predicts = eval_from_model(config,names,dataset)

        print('accuracy: ',acc)
        predict_labels.append(predicts)

    predict_labels = np.array(predict_labels,ndmin=2).T
    labels = np.array(dataset)[:,2].astype(np.int)
    
    result = [] 
    for i in range(0,len(predict_labels)):
        if sum(predict_labels[i])/len(predict_labels[i]) > 0.5:
            result.append(1)
        else:
            result.append(0)

    result = np.array(result)
    
    print('acc: ',accuracy_score(labels, result))
    
    return result, labels

def voting(config):
    dataset = read_data(config['test_data_path'])

    result, _ = vote(config, dataset)

def main(config = None):
    root_dir = "/gdrive/My Drive/electra"
    cache_dir = os.path.join(root_dir, "cache")
    output_dir = os.path.join(root_dir, "output")

    if(not os.path.exists(cache_dir)):
        os.makedirs(cache_dir)
    if (not os.path.exists(output_dir)):
        os.makedirs(output_dir)

    config = {"mode":"train",
              "root_dir": root_dir,
              "train_model_path": os.path.join(root_dir, "pretrained_model","wordpiece_small"),
              "train_data_path":os.path.join(root_dir, "data", "ratings_train_2019.txt"),
              "test_data_path": os.path.join(root_dir, "data", "ratings_test_2019.txt"),
              "test_model_path": os.path.join(root_dir, "output", "CNN-biLSTM"),
              "model_names": ['checkpoint-2344','checkpoint-2582','checkpoint-8296','checkpoint-9844','checkpoint-28128'],
              "save_file": True,
              "cache_dir_path":cache_dir,
              "model_dir_path":output_dir,
              "checkpoint":9170,
              "epoch":2,
              "learning_rate":5e-5,
              "warmup_steps":0,
              "max_grad_norm":1.0,
              "batch_size":128,
              "max_length":60,
              "num_labels":2,
              "gradient_accumulation_steps":1 #이부분 수정!
    }

    if (config["mode"] == "train"):
        train(config)
    elif (config["mode"]=="voting"):
        voting(config)
    else:
        test(config)

if(__name__=="__main__"):
    main()
