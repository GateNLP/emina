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

def download_embeddings(url='http://www.derczynski.com/sheffield/resources/2014.20M.tok.vectors.25.txt.bz2'):
    import os 
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


class Embprox:

    sim_thresh = 0.45

    open_a = r'<a href="[^"]+">'
    close_a = r'</a>'
    kw = {}
    emb = None

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
    
    matched = True # append 'matched' to output filename?

    featuregen = Embprox()
    featuregen.load_keywords('keywords')
    embeddingsfilename = '2014.20M.tok.vectors.25.txt'
    if not os.path.isfile(embeddingsfilename):
        download_embeddings()
    featuregen.load_embeddings(embeddingsfilename)

    letterstring = 'ADFGIJMNY'
    if len(sys.argv) > 1:
        letterstring = sys.argv[1]

    for letter in letterstring:
        positive = open('actionability.'+letter+'.train')
        negative_filename = 'actionability.'+letter+'.negative'
        if matched:
            negative_filename += '.matched'
        negative = open(negative_filename)

        positive_tokens = list(map(featuregen.process_text, positive))
        negative_tokens = list(map(featuregen.process_text, negative))

        outfilename = 'actionability.'+letter+'.features.sim'+str(featuregen.sim_thresh).replace('.', '_')
        if not matched:
            outfilename += '.unmatched'
        outfile = open(outfilename, 'w')

        for wordlist in positive_tokens:
            weights = featuregen.wordlist2weights(wordlist, letter)
            print('1.0 ' + ' '.join(['%0.3f' % w for w in weights]), file=outfile)
        for wordlist in negative_tokens:
            weights = featuregen.wordlist2weights(wordlist, letter)
            print('0.0 ' + ' '.join(['%0.3f' % w for w in weights]), file=outfile)

        outfile.close()
