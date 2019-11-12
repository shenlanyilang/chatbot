# -*- coding:utf-8 -*-
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from scipy.spatial.distance import cosine
import numpy as np
import jieba
import re
from collections import Counter
from spherecluster import SphericalKMeans
from embedding import Embedding
from utils import read_json, write_json


regex = re.compile(r'^[\d\w]+$')

class Retrieve(object):
    def __init__(self, path):
        self._data = pd.read_csv(path)
        self._gen_vocab()

    def _gen_vocab(self):
        word_counter = Counter()
        for sent in self._data['question']:
            word_counter.update(
                [word for word in jieba.cut(sent) if regex.search(word)])
        write_json(word_counter, './data/vocab.json')
        return word_counter

    def _init_search(self):
        self.tfidf_vectorizer = TfidfVectorizer(max_features=10000)
        tfidf = self.tfidf_vectorizer.fit_transform(self._data['qs_processed']
                                                    .tolist())
        self.word2id = self.tfidf_vectorizer.vocabulary_
        self.id2word = {idx: word for word, idx in self.word2id.items()}
        self.tfidf_trans = tfidf.toarray().T

    def _preprocess(self):
        self._data['qs_processed'] = self._data['question'].apply(lambda x: ' '.join(jieba.cut(x)))
        self.embedding = Embedding('./data/vocab.json', './data/sgns.wiki.word2vec')
        self._data['qid'] = self._data['qid'].astype(int)
        self._data['qs_embed'] = self._data['qs_processed'].apply(
            lambda x: self.embedding.sentence_embedding(x.split()))

    def _init_match(self):
        skm = SphericalKMeans(n_clusters=120, init='k-means++', n_init=20)
        data = self._data
        data = data[data['qs_embed'].apply(
            lambda x: True if np.linalg.norm(x) > 0 else False)]
        skm.fit(data['qs_embed'].tolist())
        data['skm_label'] = skm.labels_
        data = data[['qid', 'skm_label']]
        self._data = pd.merge(self._data, data, how='left', on=['qid'])
        self._data['skm_label'] = self._data['skm_label'].fillna(-1)
        self._cluster_centers = skm.cluster_centers_

    def init(self):
        print('start init ...')
        self._preprocess()
        self._init_search()
        self._init_match()
        print('finished init ...')

    def search(self, words: List[str], top=10):
        docs = set()
        for word in words:
            if word not in self.word2id:
                continue
            idx = self.word2id[word]
            docs.update(set(np.where(self.tfidf_trans[idx])[0]))
        qs_tfidf = self.tfidf_vectorizer.transform([' '.join(words)]).toarray()\
            .flatten()
        distances = []
        for idx in docs:
            vec = self.tfidf_trans[:, idx].flatten()
            dist = cosine(qs_tfidf, vec)
            distances.append((idx, dist))
        docs_ids = [idx for idx, dist in sorted(distances, key=lambda x: x[1])[:
                top]]
        qids = []
        for row in self._data.loc[docs_ids].itertuples():
            qids.append(row.qid)
        return qids

    def shallow_match(self, words: List[str], top=10):
        words_embd = self.embedding.sentence_embedding(words)
        if np.linalg.norm(words_embd) == 0:
            return []
        min_dist = 20
        clus = -1
        for cid, center_embd in enumerate(self._cluster_centers):
            dist = cosine(center_embd, words_embd)
            if dist < min_dist:
                min_dist = dist
                clus = cid
        distances = []
        for row in self._data[self._data['skm_label'] == clus].itertuples():
            qid = row.qid
            qs_embed = row.qs_embed
            distances.append((qid, cosine(qs_embed, words_embd)))
        return [qid for qid, dist in sorted(distances, key=lambda x: x[1])[:top]]
