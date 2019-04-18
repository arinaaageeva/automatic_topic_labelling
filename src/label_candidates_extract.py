import math

from itertools import chain
from collections import Counter

from tqdm import tqdm

class LabelCandidatesExtracter:
    
    '''
    This class is designed to extract label candidates from a text corpus.
    '''
    
    def __init__(self, path=None, min_count=1, max_len=10):
        
        '''
        Parameters
        ----------
        
        path : str (default None)
            Path to the folder in which 
            the model parameters are dumped.
            
        min_count : int (default 1)
            Candidates that are less common than this threshold 
            are not extracted from the corps.
            
        max_len : int (default 10)
            Candidates that have lenght less than this threshold 
            are not extracted from the corps.
        '''
        
        if path is None:
            
            self._min_count = min_count
            self._max_len = max_len
            
            '''
            int : The number of tokens in the corpus.
            '''
            self.total_tokens = 0
            
            '''
            Counter : Counter of sequences of tokens and their frequencies 
                     (contain sequences that occur more than `min_count` times).
            '''
            self.vocabulary_count = Counter()
            
        else:
            
            self.load(path)
            
    def _init_boundaries(self, tokens):
        
        '''
        Parameters
        ----------
        tokens : list with namedtuple Token (initial in src.encoder).
        
        Return
        ------
        list with tuple : Sequence of tuple where each tuple contain 
                          begin candidate index and end candidate index.
                          Initially label candidates is tokens.
        '''
        
        N = len(tokens)
        return list(zip(range(N), range(1, N+1)))
    
    def _get_lemma(self, tokens):
        
        '''
        Parameters
        ----------
        tokens : list with namedtuple Token (initial in src.encoder).
        
        Return
        ------
        str : Phrase where each token in lemma form.
        '''
        
        return ''.join([token.lemma+' '*token.space for token in tokens]).strip()
    
    def _crop_vocabulary_count(self, vocabulary_count):
        
        '''
        Parameters
        ----------
        vocabulary_count : Counter where 
                           key - sequences of tokens in lemma form 
                           and values - frequency.
        Return
        ------
        Counter : Counter of sequences of tokens and their frequencies 
                  (contain sequences that occur more than `min_count` times).
        '''
        
        return Counter(dict(filter(lambda x: x[1] >= self._min_count, vocabulary_count.items())))
    
    def _sig(self, tokens, separation_index):
        
        '''
        Parameters
        ----------
        tokens : list with namedtuple Token (initial in src.encoder).
        
        separation_index : int (from 1 to number of tokens minus 1)
        
        Return
        ------
        float : Significance from TopMine or 
                -inf if candidate isn't in `vocabulary_count`.
        '''
        
        candidate = self._get_lemma(tokens)
        candidate_count = self.vocabulary_count[candidate]
        
        if not candidate_count:
            return -float('inf')
        
        left_part = self._get_lemma(tokens[:separation_index])
        right_part = self._get_lemma(tokens[separation_index:])
            
        left_part_count = self.vocabulary_count[left_part]
        right_part_count = self.vocabulary_count[right_part]
            
        sig = (left_part_count*right_part_count)/self.total_tokens
        sig = (candidate_count - sig)/math.sqrt(candidate_count)
            
        return sig
        
    def fit(self, sentences):
        
        '''
        Frequent Chunck Mining
        
        Parameters
        ----------
        sentences : list with Sent
        
        Return
        ------
        self
        '''
        
        sentence_tokens = [sentence.tokens for sentence in sentences]
        sentence_boundaries = [self._init_boundaries(tokens) for tokens in sentence_tokens]
        
        tokens = [token.lemma for token in chain(*sentence_tokens)]
        
        self.total_tokens += len(tokens)
        self.vocabulary_count += self._crop_vocabulary_count(Counter(tokens))
        
        max_candidate_len = max([len(tokens) for tokens in sentence_tokens])
        max_candidate_len = min(max_candidate_len, self._max_len)
        
        for _ in tqdm(range(max_candidate_len)):
            
            if not sentence_boundaries:
                break
            
            chunks = []
            
            new_sentence_tokens = []
            new_sentence_boundaries = []
            
            for tokens, boundaries in zip(sentence_tokens, sentence_boundaries):
                
                new_boundaries = []
                for begin, end in boundaries[:-1]:
                    
                    chunk = self._get_lemma(tokens[begin:end])
                    
                    if chunk in self.vocabulary_count:
                        
                        new_boundaries.append((begin, end+1))
                        chunks.append(self._get_lemma(tokens[begin:end+1]))
                        
                if new_boundaries:
                    
                    new_sentence_tokens.append(tokens)
                    new_sentence_boundaries.append(new_boundaries)
                    
            sentence_tokens = new_sentence_tokens
            sentence_boundaries = new_sentence_boundaries
            
            self.vocabulary_count += self._crop_vocabulary_count(Counter(chunks))
                    
        return self
    
    def transform(self, sentence):
        
        '''
        Candidates Extraction for sentence
        
        Parameters
        ----------
        
        sentence : Sent
        
        Return
        ------
        list with tuple (begin_boundary, end_boundary, sig)
        '''
        
        candidates = []
        boundaries = self._init_boundaries(sentence.tokens)
        
        for epoch in range(len(sentence.tokens)-1):
            
            max_boundary_index = -1
            max_sig = -float('inf')
            
            for index in range(len(boundaries)-1):
                
                left_begin, left_end = boundaries[index]
                right_begin, right_end = boundaries[index+1]
                
                candidate = sentence.tokens[left_begin:right_end]
                coherence = [candidate[0].id <= token.head <= candidate[-1].id for token in candidate]
                
                if (len(candidate) > self._max_len) or (sum(coherence) != len(candidate)-1):
                    continue
                    
                sig = self._sig(candidate, left_end-left_begin)
            
                if sig > max_sig:
                    
                    max_sig = sig
                    max_boundary_index = index
            
            if max_sig < 0: break
            
            begin_boundary, _ = boundaries[max_boundary_index]
            _, end_boundary = boundaries.pop(max_boundary_index+1)
            
            boundaries[max_boundary_index] = (begin_boundary, end_boundary)

            candidates.append((begin_boundary, end_boundary, max_sig))
            
        return candidates
    
    def dump(self, path):
        
        with open(path, 'w', encoding='utf-8') as fl:
            fl.write(f'{self._min_count}\n{self._max_len}\n{self.total_tokens}\n')
            fl.write('\n'.join([f'{token}\t{count}' for token, count in self.vocabulary_count.items()]))

    def load(self, path):
        
        with open(path, 'r', encoding='utf-8') as fl:
            attributes = fl.read().split('\n')
            
        self._min_count = int(attributes[0])
        self._max_len = int(attributes[1])
        self.total_tokens = int(attributes[2])
        
        attributes = [token.split('\t') for token in attributes[3:]]
        attributes = [(token, int(count)) for token, count in attributes]
        
        self.vocabulary_count = Counter(dict(attributes))
        
        return self
    
    