from emocoder.src import utils
from pathlib import Path
import io
import numpy as np

class Embedding_Model():

    path: Path
    dims: int
    language: str

    def __getitem__(self, item):
        raise NotImplementedError("Abstract base method")

    def vocab(self):
        raise NotImplementedError


class Facebook_FastText_Vectors(Embedding_Model):

    @staticmethod
    def _load_vectors(fname, limit=None):
        fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        n, d = map(int, fin.readline().split())
        data = {}
        counter = 0
        for line in fin:
            tokens = line.rstrip().split(' ')
            data[tokens[0]] = np.array(tokens[1:], dtype=np.float32)
            if limit:  # None gets evaluated to `False`
                if counter >= limit:
                    break
                else:
                    counter += 1
        return data

    def __init__(self, limit=None):
        self.data = self._load_vectors(fname=str(self.path), limit=limit)

    def __getitem__(self, item):
        return self.data[item]

    def __contains__(self, item):
        return item in self.data

    def vocab(self):
        return list(self.data.keys())


class Facebook_CommonCrawl_English(Facebook_FastText_Vectors):
    path = utils.get_vector_dir() / "crawl-300d-2M.vec"
    language = "en"
    dims = 300

class Facebook_CommonCrawl_Spanish(Facebook_FastText_Vectors):
    path = utils.get_vector_dir() / "cc.es.300.vec"
    language = "es"
    dims = 300

class Facebook_CommonCrawl_German(Facebook_FastText_Vectors):
    path = utils.get_vector_dir() / "cc.de.300.vec"
    language = "de"
    dims = 300

class Facebook_CommonCrawl_Dutch(Facebook_FastText_Vectors):
    path = utils.get_vector_dir() / "cc.nl.300.vec"
    langauge = "nl"
    dims = 300

class Facebook_CommonCrawl_Polish(Facebook_FastText_Vectors):
    path = utils.get_vector_dir() / "cc.pl.300.vec"
    language = "pl"
    dims = 300

class Facebook_CommonCrawl_Turkish(Facebook_FastText_Vectors):
    path = utils.get_vector_dir() / "cc.tr.300.vec"
    language = "tr"
    dims = 300

EMBEDDINGS = {
    "FB_CC_EN": Facebook_CommonCrawl_English,
    "FB_CC_ES": Facebook_CommonCrawl_Spanish,
    "FB_CC_NL": Facebook_CommonCrawl_Dutch,
    "FB_CC_PL": Facebook_CommonCrawl_Polish,
    "FB_CC_TR": Facebook_CommonCrawl_Turkish,
    "FB_CC_DE": Facebook_CommonCrawl_German
}