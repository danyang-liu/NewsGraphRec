import random

news_graph_list = []

BELONG_TO_TOPIC_THRESHOLD = 5
IN_ONE_NEWS_THRESHOLD = 10
IN_ONE_USER_THRESHOLD = 10
NEWS_RELATED_RELATION_NUM = 1000

#step1 add entity-topic-belong.to.topic triple to news graph

entity_cate_num_dict = {}
fp_doc_feature = open("../../data/doc_feature.tsv", 'r', encoding='utf-8')
for line in fp_doc_feature:
    linesplit = line.split('\n')[0].split('\t')
    entities = linesplit[6].split(' ')
    category = linesplit[1]
    subcategory = linesplit[2]
    for entity in entities:
        if (entity, category) not in entity_cate_num_dict:
            entity_cate_num_dict[(entity, category)] = 1
        else:
            entity_cate_num_dict[(entity, category)] = entity_cate_num_dict[(entity, category)] + 1
        if (entity, subcategory) not in entity_cate_num_dict:
            entity_cate_num_dict[(entity, subcategory)] = 1
        else:
            entity_cate_num_dict[(entity, subcategory)] = entity_cate_num_dict[(entity, subcategory)] + 1
fp_doc_feature.close()

for key in entity_cate_num_dict:
    if entity_cate_num_dict[key] > BELONG_TO_TOPIC_THRESHOLD:
        news_graph_list.append((key[0], key[1], 'Belong.To.Topic'))

#step2 add entity-entity-in.one.news triple to news graph

entity_entity_num_dict = {}
fp_doc_feature = open("../../data/doc_feature.tsv", 'r', encoding='utf-8')
for line in fp_doc_feature:
    linesplit = line.split('\n')[0].split('\t')
    entities = linesplit[6].split(' ')
    for i in range(len(entities)-1):
        for j in range(i+1,len(entities)):
            if entities[i] != entities[j]:
                if (entities[i], entities[j]) not in entity_entity_num_dict and (entities[j], entities[i]) not in entity_entity_num_dict:
                    entity_entity_num_dict[(entities[i], entities[j])] = 1
                elif (entities[i], entities[j]) in entity_entity_num_dict:
                    entity_entity_num_dict[(entities[i], entities[j])] = entity_entity_num_dict[(entities[i], entities[j])] + 1
                else:
                    entity_entity_num_dict[(entities[j], entities[i])] = entity_entity_num_dict[(entities[j], entities[i])] + 1
fp_doc_feature.close()

for key in entity_entity_num_dict:
    if entity_entity_num_dict[key] > IN_ONE_NEWS_THRESHOLD:
        news_graph_list.append((key[0], key[1], 'In.One.News'))


#step3 add entity-entity-in.one.user triple to news graph

#put user click history in one line
fp_user_click  = open("../../data/user_history.tsv", 'r', encoding='utf-8')
user_doc_dict = {}
for line in fp_user_click:
    linesplit = line.split('\n')[0].split('\t')
    userid = linesplit[0]
    docids = linesplit[1].split(' ')
    if userid not in user_doc_dict:
        user_doc_dict[userid] = docids
fp_user_click.close()

#load doc entity dict
user_doc_entity_dict = {}
fp_doc_feature = open("../../data/doc_feature.tsv", 'r', encoding='utf-8')
for line in fp_doc_feature:
    linesplit = line.split('\n')[0].split('\t')
    docid = linesplit[0]
    entities = linesplit[6].split(' ')
    if docid not in user_doc_entity_dict:
        user_doc_entity_dict[docid] = entities
fp_doc_feature.close()

#hierarchical sample first sample doc then sample entity
entity_entity_num_dict = {}
for key in user_doc_dict:
    if len(user_doc_dict[key]) > 1 and len(user_doc_dict[key]) < 100:
        #first sample doc pair
        slected_set = set()
        doc_list = user_doc_dict[key]
        for i in range(len(doc_list)-1):
            doc_pair = random.sample(doc_list,2)
            while (doc_pair[0],doc_pair[1]) in slected_set or (doc_pair[1],doc_pair[0]) in slected_set:
                doc_pair = random.sample(doc_list, 2)
            slected_set.add((doc_pair[0],doc_pair[1]))
            #second sample entity pair
            if doc_pair[0] in user_doc_entity_dict and doc_pair[1] in user_doc_entity_dict:
                entities_doc1 = user_doc_entity_dict[doc_pair[0]]
                entities_doc2 = user_doc_entity_dict[doc_pair[1]]
                for i in range(len(entities_doc1)):
                    if entities_doc1[0] != '' and entities_doc2[0] != '':
                        entity1 = entities_doc1[i]
                        entity2 = random.sample(entities_doc2, 1)[0]
                        if (entity1, entity2) not in entity_entity_num_dict and (entity2, entity1) not in entity_entity_num_dict:
                            entity_entity_num_dict[(entity1, entity2)] = 1
                        elif (entity1, entity2) in entity_entity_num_dict:
                            entity_entity_num_dict[(entity1, entity2)] = entity_entity_num_dict[(entity1, entity2)] + 1
                        else:
                            entity_entity_num_dict[(entity2, entity1)] = entity_entity_num_dict[(entity2, entity1)] + 1
for key in entity_entity_num_dict:
    if entity_entity_num_dict[key] > IN_ONE_USER_THRESHOLD:
        news_graph_list.append((key[0], key[1], 'In.One.User'))

#step4 add news-related kg triple to news graph

news_entity_fp = open("../../data/news_entity.txt", 'r', encoding='utf-8')
news_entity_set = set()
for line in news_entity_fp:
    linesplit = line.split('\n')[0].split('\t')
    news_entity_set.add(linesplit[0])
news_entity_fp.close()

kg = []
fp_kg = open("../../data/kg.txt", 'r', encoding='utf-8')
for line in fp_kg:
    linesplit=line.split('\n')[0].split('\t')
    head = linesplit[0]
    relation = linesplit[1]
    tail = linesplit[2]
    kg.append((head, tail, relation))
fp_kg.close()

adj_dict_1 = {}
adj_dict_2 = {}
relation_count_dict = {}
kg_graph_fp = open("../../data/kg.txt", 'r', encoding='utf-8')
for line in kg_graph_fp:
    linesplit=line.split('\n')[0].split('\t')
    head = linesplit[0]
    relation  = linesplit[1]
    tail = linesplit[2]
    if head not in adj_dict_1:
        adj_dict_1[head] = set()
    else:
        adj_dict_1[head].add((relation, tail))
    if tail not in adj_dict_2:
        adj_dict_2[tail] = set()
    else:
        adj_dict_2[tail].add((relation, head))
    if relation not in relation_count_dict:
        relation_count_dict[relation] = 0
kg_graph_fp.close()

news_entity_not_head_count = 0
for entity in news_entity_set:
    news_entity_not_flag = 0
    if entity in adj_dict_1:
        news_entity_not_flag = 1
        for item in adj_dict_1[entity]:  # item (relation, tail)
            if item[1] in news_entity_set:
                relation_count_dict[item[0]] = relation_count_dict[item[0]] + 1  # add 1 hop
            if item[1] in adj_dict_1:
                for item_2 in adj_dict_1[item[1]]:
                    if item_2[1] in news_entity_set:
                        relation_count_dict[item[0]] = relation_count_dict[item[0]] + 0.1  # add 2 hop
                        relation_count_dict[item_2[0]] = relation_count_dict[item_2[0]] + 0.1  # add 2 hop

    if entity in adj_dict_2:
        news_entity_not_flag = 1
        for item in adj_dict_2[entity]:  # item (relation, tail)
            if item[1] in news_entity_set:
                relation_count_dict[item[0]] = relation_count_dict[item[0]] + 1  # add 1 hop
            if item[1] in adj_dict_2:
                for item_2 in adj_dict_2[item[1]]:
                    if item_2[1] in news_entity_set:
                        relation_count_dict[item[0]] = relation_count_dict[item[0]] + 0.1  # add 2 hop
                        relation_count_dict[item_2[0]] = relation_count_dict[item_2[0]] + 0.1  # add 2 hop

    if news_entity_not_flag == 0:
        news_entity_not_head_count = news_entity_not_head_count + 1
sorted_relation_list  = sorted(relation_count_dict.items(), key = lambda item : item[1], reverse = True)

keep_relation_set = set()
for i in range(NEWS_RELATED_RELATION_NUM):
    if i < len(sorted_relation_list):
        keep_relation_set.add(sorted_relation_list[i][0])

for triple in kg:
    if triple[2] in keep_relation_set:
        news_graph_list.append((triple[0], triple[1], triple[2]))

fp_news_graph = open("../../data/news_graph.txt", 'w', encoding='utf-8')
for item in news_graph_list:
    fp_news_graph.write(item[0]+'\t'+item[1]+'\t'+item[2]+'\n')
fp_news_graph.close()
