import torch
import random
import numpy as np

def get_vec(vec):
    vec_list = []
    vec_split1 = vec.split()
    for i in range(len(vec_split1)):
        vec_list.append(float(vec_split1[i]))
    return vec_list


def construct_embedding(entity_embedding_file):
    print('constructing embedding ...')
    entity_embedding = []
    fp_entity_embedding = open(entity_embedding_file, 'r', encoding='utf-8')
    for line in fp_entity_embedding:
        linesplit = line.split('\n')[0].split('\t')[:-1]
        linesplit = [float(i) for i in linesplit]
        entity_embedding.append(linesplit)
    return torch.FloatTensor(entity_embedding)

def construct_train(train_file, args):
    print('constructing train ...')
    train_data = {}
    user_id = []
    news_id = []
    label = []
    UV = []
    DV = []
    fp_train = open(train_file, 'r', encoding='utf-8')
    train_index = 0
    for line in fp_train:
        linesplit = line.split('\n')[0].split('\t')
        if train_index % args.negative_num == 0:
            user_id.append(linesplit[0])
            news_id.append([])
            label.append([])
            news_id[-1].append(linesplit[1])
            label[-1].append(float(linesplit[2]))
            DV.append([])
            DV[-1].append(get_vec(linesplit[3]))
            UV.append([])
            UV[-1].append(get_vec(linesplit[4]))
        else:
            news_id[-1].append(linesplit[1])
            label[-1].append(float(linesplit[2]))
            DV[-1].append(get_vec(linesplit[3]))
        train_index = train_index + 1

    train_data['user_id'] = user_id
    train_data['news_id'] = news_id
    train_data['label'] = label
    train_data['UV'] = UV
    train_data['DV'] = DV
    return train_data

def construct_test(test_file):
    print('constructing test ...')
    test_data = {}
    session_id = []
    user_id = []
    news_id = []
    label = []
    DV = []
    UV = []
    fp_test = open(test_file, 'r', encoding='utf-8')
    for line in fp_test:
        linesplit = line.split('\n')[0].split('\t')
        session_id.append(linesplit[0])
        user_id.append(linesplit[1])
        news_id.append(linesplit[2])
        label.append(float(linesplit[3]))
        DV.append(get_vec(linesplit[4]))
        UV.append(get_vec(linesplit[5]))
    test_data['session_id'] = session_id
    test_data['user_id'] = user_id
    test_data['news_id'] = news_id
    test_data['label'] = label
    test_data['DV'] = DV
    test_data['UV'] = UV
    return test_data

def construct_doc_feature(doc_feature_file, news_entity_num):
    print('constructing doc feature ...')
    doc_feature_dict = {}
    fp_doc_feature = open(doc_feature_file, 'r', encoding='utf-8')
    for line in fp_doc_feature:
        entityid_list = []
        linesplit = line.split('\n')[0].split('\t')
        docid = linesplit[0]
        entityids = linesplit[1].split(' ')
        if len(entityids) < news_entity_num:
            for i in range(0,len(entityids)):
                entityid_list.append(int(entityids[i]))
            for i in range(len(entityids), news_entity_num):
                entityid_list.append(0)
        else:
            for i in range(0, news_entity_num):
                entityid_list.append(int(entityids[i]))
        doc_feature_dict[docid] = entityid_list
    fp_doc_feature.close()
    return doc_feature_dict

def load_data(args):

    entity_embedding_file = args.rootpath + args.task + "/entity_embedding.vec"
    entity_embedding = construct_embedding(entity_embedding_file)

    train_file = args.rootpath + args.task + "/train.txt"
    train_data = construct_train(train_file, args)

    test_file = args.rootpath + args.task + "/test.txt"
    test_data = construct_test(test_file)

    doc_feature_file = args.rootpath + args.task + "/doc_feature.txt"
    doc_feature_dict = construct_doc_feature(doc_feature_file, args.news_entity_num)

    print('constructing data finishced ...')

    return entity_embedding, doc_feature_dict, train_data, test_data