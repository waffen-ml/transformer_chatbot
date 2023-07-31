from tqdm import tqdm
from torch import tensor
from torch.utils.data import Dataset
import re
import nltk
import json
import os

START = '<start>'
END = '<end>'
PAD =  '<pad>'

nltk.download('popular')


class LanguageProcessor:
    def __init__(self):
        self.lemmatizer = nltk.stem.WordNetLemmatizer()

    def _replace(self, s, selector, replacement):
        while match := re.search(selector, s):
            s_l = list(s)
            start, end = match.span()
            s_l[start:end] = replacement
            s = ''.join(s_l)
        return s

    def extract(self, s):
        s = s.lower().replace('’', "'")

        words = nltk.word_tokenize(s)
        tags = nltk.pos_tag(words)
        extracted = []

        for word, tag in tags:
            if tag == 'JJR':
                to_add = [
                    'more', self.lemmatizer.lemmatize(
                        word, 'a'
                    )
                ]
            elif tag == 'JJS':
                lemmatized = self.lemmatizer.lemmatize(
                    word, 'a'
                )
                if lemmatized == word:
                    to_add = [word]
                else:
                    to_add = ['most', lemmatized]
            elif tag[:2] == 'NN' and tag[-1] == 'S':
                to_add = [
                    self.lemmatizer.lemmatize(word)
                ]
            elif tag in ['VBP', 'VBZ', 'VBG']:
                to_add = [
                    self.lemmatizer.lemmatize(word, 'v')
                ]
            elif tag in ['VBD']:
                to_add = [
                    'did', self.lemmatizer.lemmatize(word, 'v')
                ]
            else:
                to_add = [word]

            extracted += to_add

        return extracted
