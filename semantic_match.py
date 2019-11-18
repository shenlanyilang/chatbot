# -*- coding:utf-8 -*-
from bert_serving.client import BertClient
from typing import List,Tuple
from scipy.spatial.distance import cosine
from recall.retrieve import Retrieve
from config import config
import jieba
import pandas as pd
from zhidao_spider import Spider


class ChatBot(object):
    def __init__(self):
        self._data = pd.read_csv(config['knowledge_base'])
        self._data.dropna(inplace=True)
        self._data['qid'] = self._data['qid'].astype(int)
        self.retriever = Retrieve(config['knowledge_base'])
        self.retriever.init()
        self.matcher = SemanticMatch()
        self.spider = Spider()

    def anwser(self, question):
        words = list(jieba.cut(question))
        candidates_qids = self.retriever.search(words, top=10) + self.retriever.shallow_match(words, top=10)
        cands_qs = []
        for row in self._data[self._data['qid'].isin(candidates_qids)].itertuples():
            cands_qs.append((row.qid, row.question))
        matched_qs = self.matcher.match(question, cands_qs)
        suggest_qs_ans = (None, None)
        if len(matched_qs) > 0 and matched_qs[0][1] >= config['similarity_threshold']:
            ans = self._qid2col(matched_qs[0][0], 'answer')
            return ans
        elif len(matched_qs) > 0 and matched_qs[0][1] < config['similarity_threshold']:
            qid = matched_qs[0][0]
            suggest_qs_ans = (self._qid2col(qid, 'question'), self._qid2col(qid, 'answer'))
        spider_qs_ans = self.spider.search_qs(question)
        spider_qs_nb = [(i, spider_qs_ans[i][0]) for i in range(len(spider_qs_ans))]
        spider_match = self.matcher.match(question, spider_qs_nb)
        if len(spider_match) > 0 and spider_match[0][1] >= config['similarity_threshold']:
            return spider_qs_ans[spider_match[0][0]][1]
        elif len(spider_match) > 0 and spider_match[0][1] < config['similarity_threshold'] and suggest_qs_ans == (None, None):
            ind = spider_match[0][0]
            suggest_qs_ans = (spider_qs_ans[ind][0], spider_qs_ans[ind][1])
        if suggest_qs_ans != (None, None):
            return config['suggest_temp'].format(question=suggest_qs_ans[0], answer=suggest_qs_ans[1])
        else:
            return self._gen_suggest_qs(question)

    def _gen_suggest_qs(self, question):
        return '非常抱歉，我不太明白您的意思'

    def _qid2col(self, qid, col):
        return self._data[self._data['qid'] == qid][col].tolist()[0]


class SemanticMatch(object):
    """
    基于预训练中文bert的深层语义匹配
    """
    def __init__(self):
        self.bc = BertClient()

    def match(self, question:str, candidates:List[Tuple[int, str]], top=3):
        """
        candidates是召回的候选问题list，list中是(pid,question)组成的tuple
        :param question:
        :param candidates:
        :param top:
        :return:
        """
        qids, cand_sents = zip(*candidates)
        qs_embed = self.bc.encode([question])[0]
        cand_embed = self.bc.encode(list(cand_sents))
        similarities = [1. - cosine(qs_embed, embd) for embd in cand_embed]
        qids_sim = list(zip(qids, similarities))
        return sorted(qids_sim, key=lambda x: x[1], reverse=True)[:top]

if __name__ == '__main__':
    qs = '网银贷款申请的种类有哪些'
    cands = [(1, '一个客户最多可以用几个手机号码开通账户变动通知服务'), (2, '车票允许改签吗'),
             (3, '补发网银盾'), (4, '代发工资	')]

    sm = SemanticMatch()
    result = sm.match(qs, cands)
    print(result)