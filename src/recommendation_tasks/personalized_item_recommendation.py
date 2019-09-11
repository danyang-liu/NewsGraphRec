import torch
import torch.nn as nn

class Softmax_BCELoss(nn.Module):
    def __init__(self, args):
        super(Softmax_BCELoss, self).__init__()
        self.args = args
        self.softmax = nn.Softmax()
        self.bceloss = nn.BCELoss()

    def forward(self, predict, truth):
        predict = self.softmax(predict)
        loss = self.bceloss(predict, truth)
        return loss


class NewsGraphRec(nn.Module):

    def __init__(self, args, doc_feature_dict, entity_embedding):
        super(NewsGraphRec, self).__init__()
        self.args = args
        self.doc_feature_dict = doc_feature_dict
        self.entity_embedding = entity_embedding

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=0)
        self.mlp_layer1 = nn.Linear(2*self.args.embedding_dim, self.args.layer_dim)
        self.mlp_layer2 = nn.Linear(self.args.layer_dim, 1)
        self.attention_embedding_layer1 = nn.Linear(self.args.embedding_dim,self.args.layer_dim)
        self.attention_embedding_layer2 = nn.Linear(self.args.layer_dim,1)
        self.transformlayer = nn.Linear(2*self.args.embedding_dim, self.args.embedding_dim)
        self.cos = torch.nn.CosineSimilarity(dim=-1)
        self.entity_embedding_lookup = nn.Embedding.from_pretrained(entity_embedding)

    def attention_layer(self, entity_embeddings):
        att_value1 = self.relu(self.attention_embedding_layer1(entity_embeddings))
        att_value = self.attention_embedding_layer2(att_value1)
        soft_att_value = self.softmax(att_value)
        weighted_entity_embedding = soft_att_value*entity_embeddings
        weighted_entity_embedding_sum = torch.sum(weighted_entity_embedding, dim=-2)
        return weighted_entity_embedding_sum


    def get_entities_ids(self, news_id):
        entities = []
        for news in news_id:
            if type(news) == str:
                entities.append(self.doc_feature_dict[news])
            else:
                entities.append([])
                for news_i in news:
                    entities[-1].append(self.doc_feature_dict[news_i])
        return entities

    def forward(self, news_id, dv, uv):
        user_vector = torch.squeeze(torch.FloatTensor(uv).cuda())
        doc_vector = torch.squeeze(torch.FloatTensor(dv).cuda())

        entity_ids = self.get_entities_ids(news_id)
        entity_embeddings = self.entity_embedding_lookup(torch.tensor(entity_ids).cuda())
        attentive_entity_embedding = self.attention_layer(entity_embeddings)

        new_doc_embedding = self.relu(self.transformlayer(torch.cat([attentive_entity_embedding, doc_vector] , dim=-1)))

        if len(new_doc_embedding.shape) > len(user_vector.shape):
            user_vector = torch.unsqueeze(user_vector, 1)
            user_vector = user_vector.expand(user_vector.shape[0], new_doc_embedding.shape[1],
                                             user_vector.shape[2])

        u_n_embedding = torch.cat([user_vector, new_doc_embedding], dim=(len(user_vector.shape) - 1))
        feature_embedding = self.relu(self.mlp_layer1(u_n_embedding))
        predict = self.sigmoid(self.mlp_layer2(feature_embedding))

        return predict