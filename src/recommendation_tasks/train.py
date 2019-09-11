from personalized_item_recommendation import NewsGraphRec
from personalized_item_recommendation import Softmax_BCELoss
import torch
from torch import optim
from utils.metric import *
import numpy as np

def train_test(args, data):
    entity_embedding = data[0]
    doc_feature_dict = data[1]
    train_data = data[2]
    test_data = data[3]

    model = NewsGraphRec(args, doc_feature_dict, entity_embedding).cuda()

    criterion = Softmax_BCELoss(args)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.l2_regular)

    for epoch in range(args.epoch):

        start_list = list(range(0, len(train_data['label']), args.batch_size))
        np.random.shuffle(start_list)
        total_loss = 0
        model.train()
        for start in start_list:
            end = start + args.batch_size
            out = model(train_data['news_id'][start:end],  train_data['DV'][start:end], train_data['UV'][start:end])
            loss = criterion(out, torch.tensor(train_data['label'][start:end]).cuda())
            total_loss = total_loss + loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('epoch {} loss {}'.format(epoch, total_loss))

        model.eval()
        y_pred = []
        start_list = list(range(0, len(test_data['label']), args.batch_size))
        for start in start_list:
            if start + args.batch_size <= len(test_data['label']):
                end = start + args.batch_size
            else:
                end = len(test_data['label'])
            out = model(test_data['news_id'][start:end],
                        test_data['DV'][start:end], test_data['UV'][start:end]).view(end - start).cpu().data.numpy()
            y_pred = y_pred + out.tolist()
            pass

        truth = test_data['label']
        score = roc_auc_score(truth, y_pred)
        sess_id = test_data['session_id']
        ndcg = []
        truth_i = []
        pred_i = []
        for i in range(len(truth)):
            truth_i.append(truth[i])
            pred_i.append(y_pred[i])
            if i + 1 == len(truth):
                ndcg.append(cal_ndcg_float(truth_i, pred_i, 10))
                truth_i = []
                pred_i = []
            elif sess_id[i] != sess_id[i + 1]:
                ndcg.append(cal_ndcg_float(truth_i, pred_i, 10))
                truth_i = []
                pred_i = []
        ndcg = np.mean(np.array(ndcg))
        print('epoch:%d AUC:%.6f NDCG:%.6f' % (epoch + 1, score, ndcg))
