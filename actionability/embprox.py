#!/usr/bin/env python3

# representation:
#  for each keyword
#  what % of words in the sentence come within boundary distance of that keyword?
# this gives a vector
# infer w/ ... SVC?

import nltk
import numpy as np
import re
import sys

from gensim.models.keyedvectors import KeyedVectors
sys.path.append('twokenize/')
from twokenize_wrapper import tokenize

default_embeddings_file = '2014.20M.tok.vectors.25.txt'


class Embprox:

    sim_thresh = 0.45

    open_a = r'<a href="[^"]+">'
    close_a = r'</a>'
    kw = {}
    emb = None

    def download_embeddings(url='http://www.derczynski.com/sheffield/resources/'+default_embeddings_file+'.bz2'):
        import requests
        print('downloading from', url)
        r = requests.get(url, allow_redirects=True)
        dest = url.split('/')[-1]
        open(dest, 'wb').write(r.content)
        if url.endswith('.bz2'):
            import bz2
            b = bz2.BZ2File(dest)
            f = open(dest.replace('.bz2', ''), 'wb')
            f.write(b.read())
            f.close()
            b.close()
            os.unlink(dest)

    def process_text(self, sentence):
        sentence = sentence.strip()
        sentence = re.sub(self.open_a, '', sentence)
        sentence = re.sub(self.close_a, '', sentence)

        return tokenize(sentence.lower())

    def unison_shuffled_copies(self, a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]

    def wordlist2weights(self, wordlist, letter): 
        weights = []
        nwords = len(wordlist)
        for refword in self.kw[letter]:
            weight = 0.0
            for word in wordlist: # per keyword
                try:
                    emb_sim = self.emb.similarity(word,refword)
                    if emb_sim > self.sim_thresh:
                        weight += emb_sim/nwords # find the proportion of similar words in the candidate text, scale by similarity
                except KeyError:
                    pass
            weights.append(weight)
        return weights

    def load_keywords(self, keywordfilename):
        kwfile = open(keywordfilename, 'r')
        self.kw = {}
        for line in kwfile:
            r = line.strip().split()
            self.kw[r[0]] = r[1:]

    def load_embeddings(self, embeddingsfilename):
        self.emb = KeyedVectors.load_word2vec_format(embeddingsfilename, binary=False)

if __name__ == "__main__":
    
    import os 
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--matched', help='Assume matched positive and negative examples, with .matched filename suffix', action='store_true')
    parser.add_argument('--embeddings', help='Embeddings file name (in text format)', default=default_embeddings_file, action='store')
    parser.add_argument('--classes', help='Which classe(s) should be drawn from?', default='ADFGIJMNY')
    parser.add_argument('--prefix', help='The filename prefix for reading from', default='actionability')
    parser.add_argument('--threshold', help='Cosine similarity threshold for considering terms', default=0.45, type=float)
    parser.add_argument('--keywords', help='Keywords file', default='keywords')
    opts = parser.parse_args()

    featuregen = Embprox()
    featuregen.load_keywords('keywords')
    if not os.path.isfile(opts.embeddings) and opts.embeddings == default_embeddings_file:
        featuregen.download_embeddings()
    featuregen.load_embeddings(opts.embeddings)

    featuregen.sim_thresh = opts.threshold

    for letter in opts.classes:
        positive = open(opts.prefix+'.'+letter+'.train')
        negative_filename = opts.prefix+'.'+letter+'.negative'
        if opts.matched:
            negative_filename += '.matched'
        negative = open(negative_filename)

        positive_tokens = list(map(featuregen.process_text, positive))
        negative_tokens = list(map(featuregen.process_text, negative))

        outfilename = opts.prefix+'.'+letter+'.features.sim'+str(featuregen.sim_thresh).replace('.', '_')
        if not opts.matched:
            outfilename += '.unmatched'
        outfile = open(outfilename, 'w')

        for wordlist in positive_tokens:
            weights = featuregen.wordlist2weights(wordlist, letter)
            print('1.0 ' + ' '.join(['%0.3f' % w for w in weights]), file=outfile)
        for wordlist in negative_tokens:
            weights = featuregen.wordlist2weights(wordlist, letter)
            print('0.0 ' + ' '.join(['%0.3f' % w for w in weights]), file=outfile)

        outfile.close()
