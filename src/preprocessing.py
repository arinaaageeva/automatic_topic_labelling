import re

from typing import List
from rusenttokenize import ru_sent_tokenize
from spacy.lang.ru import Russian
from spacy_russian_tokenizer import RussianTokenizer, MERGE_PATTERNS, SYNTAGRUS_RARE_CASES
from pyaspeller import Word, YandexSpeller
from transliterate import translit
from rnnmorph.predictor import RNNMorphPredictor
from ufal.udpipe import Model, Pipeline, ProcessingError
from collections import Counter
from functools import reduce

from sklearn.base import BaseEstimator, TransformerMixin

class Token:
    
    def __init__(self, \
                 token_id: int, \
                 form: str, \
                 lemma: str = None, \
                 upos: str = None, \
                 xpos: str = None, \
                 feats: str = None, \
                 head: int = None, \
                 deprel: str = None, \
                 deps: str = None, \
                 space: bool = None):
        
        self.token_id = token_id
        self.form = form
        self.lemma = lemma
        self.upos = upos
        self.xpos = xpos
        self.feats = feats
        self.head = head
        self.deprel = deprel
        self.deps = deps
        self.space = space

class Sentence:
    
    def __init__(self, \
                 sent_id: int, \
                 text: str, \
                 tokens: List[Token] = None):
        
        self.sent_id = sent_id
        self.text = text
        self.tokens = tokens
    
class ReplaceChars(BaseEstimator, TransformerMixin):
    
    def __init__(self, chars_map=None):
        self.chars_map = chars_map
    
    def fit(self, X, y=None):
        return self
    
    def replace_chars(self, x):
        for replaceable_char, replacement_char in self.chars_map.items():
            x = x.replace(replaceable_char, replacement_char)
        return x
    
    def transform(self, X):
        if self.chars_map:
            X = [self.replace_chars(x) for x in X]
        return X
    
class RegExprSub(BaseEstimator, TransformerMixin):
    
    def __init__(self, pattern=None, repl=None):
        
        if pattern:
            pattern = re.compile(pattern)
        
        self.pattern = pattern
        self.repl = repl
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if self.pattern and self.repl:
            X = [self.pattern.sub(self.repl, x) for x in X]
        return X
        
class Strip(BaseEstimator, TransformerMixin):
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return [x.strip() for x in X]

#https://github.com/deepmipt/ru_sentence_tokenizer 
class RusSentTokenizer(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        return self
    
    def sent_tokenize(self, x):
        return [Sentence(sent_id, sent) 
                for sent_id, sent in 
                enumerate(ru_sent_tokenize(x))]
    
    def transform(self, X):
        return [self.sent_tokenize(x) for x in X]
    
class Yandex_Speller(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        self.speller = YandexSpeller()
    
    def fit(self, X, y=None):
        return self
    
    def sent_spell(self, sent):
        
        changes = {change['word']: change['s'][0] 
                   for change in self.speller.spell(sent.text) 
                   if change['s']}
        
        for word, suggestion in changes.items():
            sent.text = sent.text.replace(word, suggestion)
        
        return sent
    
    def spell(self, x):
        
        x_spell = []
        for sent in x:
            try:
                sent = self.sent_spell(sent)
                x_spell.append(sent)
            except:
                pass
                
        return x_spell
    
    def transform(self, X):
        return [self.spell(x) for x in X]
    
#https://github.com/aatimofeev/spacy_russian_tokenizer
class Spacy_RusWordTokenizer(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        
        self.rus_word_tokenizer = Russian()
        pipe = RussianTokenizer(self.rus_word_tokenizer, MERGE_PATTERNS + SYNTAGRUS_RARE_CASES)
        self.rus_word_tokenizer.add_pipe(pipe, name='russian_tokenizer')
    
    def fit(self, X, y=None):
        return self
    
    def sent_tokenize(self, sent):
        sent.tokens = [Token(token_id, token.text) 
                       for token_id, token in 
                       enumerate(self.rus_word_tokenizer(sent.text), 1)]
        return sent
    
    def tokenize(self, x):
        return [self.sent_tokenize(sent) for sent in x]
    
    def transform(self, X):
        return [self.tokenize(x) for x in X]
    
class SpaceDetecter(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        return self
    
    def sent_detect(self, sent):
        
        spaces = []
        viewed_text = sent.text
        
        for token in sent.tokens:
            
            index = viewed_text.find(token.form)
            spaces.append(bool(index))
            
            viewed_text = viewed_text[index+len(token.form):]
        
        spaces = spaces[1:]+[False]
        for token, space in zip(sent.tokens, spaces):
            token.space = space
            
        return sent
    
    def detect(self, x):
        return [self.sent_detect(sent) for sent in x]
    
    def transform(self, X):
        return [self.detect(x) for x in X]

#https://github.com/IlyaGusev/rnnmorph
class MorphPredictor(BaseEstimator, TransformerMixin):
    
    def __init__(self, translit_flag=True):
        self.translit_flag = translit_flag
        self.rnnmorph = RNNMorphPredictor(language='ru')
        
    def fit(self, X, y=None):
        return self
    
    def translit(self, form):
        return translit(form, 'ru') if re.match(r'[a-zA-Z]+', form) else form
        
    def sent_predict(self, sent):
        
        forms = [token.form for token in sent.tokens]
        if self.translit_flag:
            forms = [self.translit(form) for form in forms]
        forms = self.rnnmorph.predict(forms)
        
        for token, form in zip(sent.tokens, forms):
            token.lemma = form.normal_form
            token.upos = form.pos
            token.feats = form.tag
        
        return sent
    
    def predict(self, x):
        return [self.sent_predict(sent) for sent in x]
    
    def transform(self, X):
        return [self.predict(x) for x in X]
    
class MorphFilter(BaseEstimator, TransformerMixin):
    
    def __init__(self, upos_set=set()):
        self.upos_set = upos_set
        
    def fit(self, X, y=None):
        return self
    
    def sent_filtration(self, sent):
        sent.tokens = [token for token in sent.tokens if token.upos in self.upos_set]
        return sent
    
    def filtration(self, x):
        return [self.sent_filtration(sent) for sent in x]
    
    def transform(self, X):
        return [self.filtration(x) for x in X] 
    
class SyntaxParser(BaseEstimator, TransformerMixin):
    
    def __init__(self, model_path):
        
        self.parser_model = Model.load(model_path)
        self.parser_pipeline = Pipeline(self.parser_model, 'conllu', Pipeline.NONE, Pipeline.DEFAULT, 'conllu')
        
    def fit(self, X, y=None):
        return self
    
    def parse(self, x):
        return self.parser_pipeline.process(x, ProcessingError())
    
    def transform(self, X):
        return [self.parse(x) for x in X]
    
class CoNLLUFormatEncoder(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        return self
    
    def token_encode(self, token):
        
        encode_nan = lambda x: '_' if x is None else x
        encode_space = lambda x: 'SpaceAfter=No' if x is False else '_'
        
        conllu_token = f'{token.token_id}\t'
        conllu_token += f'{token.form}\t'
        conllu_token += f'{encode_nan(token.lemma)}\t'
        conllu_token += f'{encode_nan(token.upos)}\t'
        conllu_token += f'{encode_nan(token.xpos)}\t'
        conllu_token += f'{encode_nan(token.feats)}\t'
        conllu_token += f'{encode_nan(token.head)}\t'
        conllu_token += f'{encode_nan(token.deprel)}\t'
        conllu_token += f'{encode_nan(token.deps)}\t'
        conllu_token += f'{encode_space(encode_nan(token.space))}'
        
        return conllu_token
        
    def sent_encode(self, sent):
        conllu_sent = f'# sent_id = {sent.sent_id}\n# text = {sent.text}\n'
        for token in sent.tokens:
            conllu_sent += f'{self.token_encode(token)}\n'
        return conllu_sent.strip()
    
    def encode(self, x):
        return '\n\n'.join([self.sent_encode(sent) for sent in x])+'\n\n'
    
    def transform(self, X):
        return [self.encode(x) for x in X]
    
class  CoNLLUFormatDecoder(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        return self
    
    def token_decode(self, token):
        
        decode_nan = lambda x: None if x is '_' else x
        decode_int = lambda x: x if x is None else int(x)
        decode_space = lambda x: True if x is None else False
        
        token = [decode_nan(item) for item in token.split('\t')]
        
        token[0] = decode_int(token[0])
        token[6] = decode_int(token[6])
        token[9] = decode_space(token[9])
        
        return Token(*token)
    
    def sent_decode(self, sent):
        
        sent = sent.split('\n')
        
        sent_id = int(sent[0][12:])
        text = sent[1][9:]
        tokens = [self.token_decode(token) for token in sent[2:]]
        
        return Sentence(sent_id, text, tokens)
    
    def decode(self, x):
        return [self.sent_decode(sent) for sent in x.strip().split('\n\n')]
    
    def transform(self, X):
        return [self.decode(x) for x in X]
    
class VowpalWabbitFormatEncoder(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        return self
    
    def sent_encode(self, sent):
        return Counter([token.lemma for token in sent.tokens])
    
    def encode(self, x):
        x = reduce((lambda x, y: x + y), [self.sent_encode(sent) for sent in x])
        return ' '.join([f'{token}' + (f':{count}' if count > 1 else '') for token, count in x.items()])
    
    def transform(self, X):
        return [self.encode(x) for x in X]
        