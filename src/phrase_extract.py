import json
import math

from tqdm import tqdm
from itertools import chain

class PhraseExtracter:
    
    def __init__(self, path=None, min_count=1, max_phrase_len=float('inf')):
        
        if path is None:
            self.total_words = 0
            self.vocabulary_count = {}

            self._min_count = min_count
            self._max_phrase_len = max_phrase_len
        
        else:
            self.load(path)
            
    def get_form(self, tokens, lemma=True):
        
        form = lambda token: token.lemma if lemma else token.token
        return ''.join([form(token)+' '*token.space for token in tokens]).strip()
            
    def fit(self, X):

        for tokens in tqdm(chain(*[[sent.tokens for sent in sentences] for sentences in X])):
            
            N = len(tokens)
            self.total_words += N
            
            max_chunk_len = min(N, self._max_phrase_len)
            
            for chunk_len in range(1, max_chunk_len):
                for index in range(max_chunk_len-chunk_len):
                    
                    chunk = self.get_form(tokens[index:index+chunk_len+1])
                    
                    self.vocabulary_count.setdefault(chunk, 0)
                    self.vocabulary_count[chunk] += 1
                    
        self.vocabulary_count = dict([(chunk, count) for chunk, count in self.vocabulary_count.items() 
                                      if count >= self._min_count])
        
        return self
    
    def sig(self, left_part_phrase, right_part_phrase, phrase):
        
        phrase_sig = self._vocabulary_sig.get(phrase)
        
        if phrase_sig is None:
            
            if phrase in self.vocabulary_count:
                        
                phrase_count = self.vocabulary_count[phrase]
                        
                left_part_phrase_count = self.vocabulary_count.get(left_part_phrase, 0)
                right_part_phrase_count = self.vocabulary_count.get(right_part_phrase, 0)
                        
                phrase_sig = (left_part_phrase_count*right_part_phrase_count)/self.total_words
                phrase_sig = (phrase_count - phrase_sig)/math.sqrt(phrase_count)
                        
            else:
                        
                phrase_sig = -float('inf')
                
            self._vocabulary_sig[phrase] = phrase_sig
            
        return phrase_sig
        
    
    def transform(self, tokens):
        
        phrases = []
        self._vocabulary_sig = {}
        
        N = len(tokens)
        boundaries = list(zip(range(N), range(1, N+1)))
        
        while N > 1:
            
            max_boundary_index = -1
            max_sig = -float('inf')
            
            for index in range(N-1):
                
                left_begin_boundary, left_end_boundary = boundaries[index]
                right_begin_boundary, right_end_boundary = boundaries[index+1]
                
                phrase = tokens[left_begin_boundary:right_end_boundary]
                phrase_coherence = [phrase[0].id <= token.head <= phrase[-1].id for token in phrase]
                
                if sum(phrase_coherence) != len(phrase)-1:
                    continue
                
                left_part_phrase = self.get_form(tokens[left_begin_boundary:left_end_boundary])
                right_part_phrase = self.get_form(tokens[right_begin_boundary:right_end_boundary])
                
                phrase_sig = self.sig(left_part_phrase, right_part_phrase, self.get_form(phrase))
                    
                if phrase_sig > max_sig:
                    max_sig = phrase_sig
                    max_boundary_index = index
                    
            begin_boundary, _ = boundaries[max_boundary_index]
            _, end_boundary = boundaries.pop(max_boundary_index+1)
            
            boundaries[max_boundary_index] = (begin_boundary, end_boundary)
            
            phrase = self.get_form(tokens[begin_boundary:end_boundary], lemma=False)
            phrase_lemma = self.get_form(tokens[begin_boundary:end_boundary])
            
            phrases.append((begin_boundary, end_boundary, phrase, phrase_lemma, max_sig))
                
            N -= 1
            
        return phrases
            
    def dump(self, path):
        
        with open(path, 'w', encoding='utf-8') as fl:
            json.dump({'total_words':self.total_words, 'vocabulary_count':self.vocabulary_count}, fl)
            
    def load(self, path):
        
        with open(path, 'r', encoding='utf-8') as fl:
            attributes = json.load(fl)
            
        self.total_words = attributes['total_words']
        self.vocabulary_count = attributes['vocabulary_count']
        
        return self