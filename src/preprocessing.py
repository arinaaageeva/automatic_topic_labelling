import re
import time

from typing import List
from rusenttokenize import ru_sent_tokenize
from spacy.lang.ru import Russian
from spacy_russian_tokenizer import RussianTokenizer, MERGE_PATTERNS, SYNTAGRUS_RARE_CASES
from deeppavlov import configs, build_model
from pyaspeller import Word, YandexSpeller
from transliterate import translit
from rnnmorph.predictor import RNNMorphPredictor
from ufal.udpipe import Model, Pipeline, ProcessingError
from collections import Counter
from functools import reduce

from sklearn.base import BaseEstimator, TransformerMixin

from tqdm import tqdm

class Token:
    
    def __init__(self, \
                 token_id: int, \
                 form: str, \
                 lemma: str = None, \
                 pos: str = None, \
                 ne: str = None, \
                 feats: str = None, \
                 head: int = None, \
                 deprel: str = None, \
                 deps: str = None, \
                 space: bool = None):
        
        self.token_id = token_id
        self.form = form
        self.lemma = lemma
        self.pos = pos
        self.ne = ne
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
        
class PreProcesser(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        
        print(type(self).__name__)
        time.sleep(1)
        
        return [self.transform_item(x) for x in tqdm(X)]
    
class Cleaner(PreProcesser):
    
    def transform_sent(self, sent):
        sent.text = self.transform_text(sent.text)
        return sent
    
    def transform_item(self, x):
        if self.sent:
            return [self.transform_sent(sent) for sent in x]
        return self.transform_text(x)
    
class ReplaceChars(Cleaner):
    
    def __init__(self, chars_map=None, sent=False):
        self.chars_map = {} if chars_map is None else chars_map
        self.sent = sent
        
    def transform_text(self, text):
        for replaceable_char, replacement_char in self.chars_map.items():
            text = text.replace(replaceable_char, replacement_char)
        return text
    
class ReplacePart(Cleaner):
    
    def __init__(self, pattern=None, rule=None, sent=False):
        self.pattern = pattern if pattern is None else re.compile(pattern)
        self.rule = (lambda x: x) if rule is None else rule
        self.sent = sent
        
    def transform_text(self, text):
        if self.pattern is not None:
            for part in self.pattern.findall(text):
                text = text.replace(part, self.rule(part))
        return text
    
class RegExprSub(Cleaner):
    
    def __init__(self, pattern=None, repl=None, sent=False):
        self.pattern = pattern if pattern is None else re.compile(pattern)
        self.repl = repl
        self.sent = sent
        
    def transform_text(self, text):
        if (self.pattern is not None) and (self.repl is not None):
            text = self.pattern.sub(self.repl, text)
        return text

class Strip(Cleaner):
    
    def __init__(self, sent=False):
        self.sent = sent
    
    def transform_text(self, text):
        return text.strip()
    
class Yandex_Speller(Cleaner):
    
    def __init__(self, sent=False):
        self.speller = YandexSpeller()
        self.sent = sent
    
    def transform_text(self, text):
        
        try:
            changes = {change['word']: change['s'][0] 
                       for change in self.speller.spell(text) if change['s']}
            for word, suggestion in changes.items():
                text = text.replace(word, suggestion)
        
        except:
            pass
        
        return text

#https://github.com/deepmipt/ru_sentence_tokenizer 
class RusSentTokenizer(PreProcesser):
    
    def transform_item(self, x):
        return [Sentence(sent_id, sent) for sent_id, sent in enumerate(ru_sent_tokenize(x), 1)]
    
#https://github.com/aatimofeev/spacy_russian_tokenizer
class Spacy_RusWordTokenizer(PreProcesser):
    
    def __init__(self):
        
        self.rus_word_tokenizer = Russian()
        pipe = RussianTokenizer(self.rus_word_tokenizer, MERGE_PATTERNS + SYNTAGRUS_RARE_CASES)
        self.rus_word_tokenizer.add_pipe(pipe, name='russian_tokenizer')
    
    def transform_text(self, text):
        return [Token(token_id, token.text) for token_id, token in enumerate(self.rus_word_tokenizer(text), 1)]
    
    def transform_sent(self, sent):
        sent.tokens = self.transform_text(sent.text)
        return sent
    
    def transform_item(self, x):
        return [self.transform_sent(sent) for sent in x]
    
#http://docs.deeppavlov.ai/en/master/features/models/ner.html    
class NER_RusWordTokenizer(PreProcesser):
    
    def __init__(self):
        self.rus_ne_recognizer = build_model(configs.ner.ner_rus_bert)
    
    def transform_item(self, x):
        
        new_x = []
        
        for sent, tokens, tokens_ne in zip(*[x]+self.rus_ne_recognizer([sent.text for sent in x])):
            sent.tokens =  [Token(token_id, token, ne=token_ne) 
                            for token_id, (token, token_ne) in enumerate(zip(tokens, tokens_ne), 1)]
            new_x.append(sent)
    
        return new_x

#https://github.com/IlyaGusev/rnnmorph
class MorphPredictor(PreProcesser):
    
    def __init__(self, translit_flag=True):
        self.translit_flag = translit_flag
        self.rnnmorph = RNNMorphPredictor(language='ru')
    
    def translit(self, form):
        return translit(form, 'ru') if re.match(r'[a-zA-Z]+', form) else form
    
    def transform_sent(self, sent):
        
        forms = [token.form for token in sent.tokens]
        if self.translit_flag:
            forms = [self.translit(form) for form in forms]
        forms = self.rnnmorph.predict(forms)
        
        for token, form in zip(sent.tokens, forms):
            token.lemma = form.normal_form
            token.pos = form.pos
            token.feats = form.tag
        
        return sent
        
    def transform_item(self, x):
        return [self.transform_sent(sent) for sent in x]
    
class SpaceDetecter(PreProcesser):
    
    def transform_sent(self, sent):
        
        spaces = []
        text = sent.text
        
        for token in sent.tokens:
            
            index = text.find(token.form)
            spaces.append(bool(index))
            
            text = text[index+len(token.form):]
            
        spaces = spaces[1:]+[False]
        
        for token, space in zip(sent.tokens, spaces):
            token.space = space
        
        return sent
    
    def transform_item(self, x):
        return [self.transform_sent(sent) for sent in x]
    
class NER_Correcter(PreProcesser):
    
    def join(self, chunks, spaces):
        return ''.join([chunk + (' ' if space else '') for chunk, space in zip(chunks, spaces)]).strip()
    
    def join_forms(self, chunks):
        return self.join([chunk.form for chunk in chunks], [chunk.space for chunk in chunks])
    
    def join_lemmas(self, chunks):
        return self.join([chunk.lemma for chunk in chunks], [chunk.space for chunk in chunks])
    
    def join_chunks(self, chunks):
        
        token_id = chunks[0].token_id
        
        form = self.join_forms(chunks)
        lemma = self.join_lemmas(chunks)
        
        pos = 'NOUN'
        ne = chunks[0].ne
        
        feats = [token.feats for token in chunks if token.pos == pos]
        feats = feats[0] if feats else 'Case=Nom|Gender=Masc|Number=Sing'
        
        space = chunks[-1].space
        
        return Token(token_id, form, lemma=lemma, pos=pos, ne=ne, feats=feats, space=space)
    
    def transform_sent(self, sent):
        
        chunks = []
        new_tokens = []
        
        for token in sent.tokens:
            
            if token.ne[0] == 'B':
                
                if chunks:
                    new_tokens.append(self.join_chunks(chunks))
                
                chunks = [token]
                
            elif token.ne[0] == 'I':
                
                if chunks and (chunks[0].ne[2:] == token.ne[2:]):
                    chunks.append(token)
                    
                else:
                    
                    if chunks:
                        new_tokens.append(self.join_chunks(chunks))
                        
                    chunks = []
                    
                    token.ne = 'O'
                    new_tokens.append(token)
                    
            else:
                
                if chunks:
                    new_tokens.append(self.join_chunks(chunks))
                        
                chunks = []
                
                new_tokens.append(token)
                
        if chunks:
            new_tokens.append(self.join_chunks(chunks))
        
        sent.tokens = new_tokens
        return sent
    
    def transform_item(self, x):
        return [self.transform_sent(sent) for sent in x]
    
class MorphFilter(PreProcesser):
    
    def __init__(self, pos_set=None):
        self.pos_set = set() if pos_set is None else pos_set
    
    def transform_sent(self, sent):
        sent.tokens = [token for token in sent.tokens if token.pos in self.pos_set]
        return sent
    
    def transform_item(self, x):
        return [self.transform_sent(sent) for sent in x]
    
# class SyntaxParser(BaseEstimator, TransformerMixin):
    
#     def __init__(self, model_path):
        
#         self.parser_model = Model.load(model_path)
#         self.parser_pipeline = Pipeline(self.parser_model, 'conllu', Pipeline.NONE, Pipeline.DEFAULT, 'conllu')
        
#     def fit(self, X, y=None):
#         return self
    
#     def parse(self, x):
#         return self.parser_pipeline.process(x, ProcessingError())
    
#     def transform(self, X):
#         return [self.parse(x) for x in X]
    
class CoNLLUFormatEncoder(PreProcesser):
    
    def transform_token(self, token):
        
        encode_nan = lambda x: '_' if x is None else x
        encode_space = lambda x: 'SpaceAfter=No' if x is False else '_'
        
        conllu_token = f'{token.token_id}\t'
        conllu_token += f'{token.form}\t'
        conllu_token += f'{encode_nan(token.lemma)}\t'
        conllu_token += f'{encode_nan(token.pos)}\t'
        conllu_token += f'{encode_nan(token.ne)}\t'
        conllu_token += f'{encode_nan(token.feats)}\t'
        conllu_token += f'{encode_nan(token.head)}\t'
        conllu_token += f'{encode_nan(token.deprel)}\t'
        conllu_token += f'{encode_nan(token.deps)}\t'
        conllu_token += f'{encode_space(encode_nan(token.space))}'
        
        return conllu_token
        
    def transform_sent(self, sent):
        conllu_sent = f'# sent_id = {sent.sent_id}\n# text = {sent.text}\n'
        for token in sent.tokens:
            conllu_sent += f'{self.transform_token(token)}\n'
        return conllu_sent.strip()
    
    def transform_item(self, x):
        return '\n\n'.join([self.transform_sent(sent) for sent in x])+'\n\n'
    
class  CoNLLUFormatDecoder(PreProcesser):
    
    def transform_token(self, token):
        
        decode_nan = lambda x: None if x is '_' else x
        decode_int = lambda x: x if x is None else int(x)
        decode_space = lambda x: True if x is None else False
        
        token = [decode_nan(item) for item in token.split('\t')]
        
        token[0] = decode_int(token[0])
        token[6] = decode_int(token[6])
        token[9] = decode_space(token[9])
        
        return Token(*token)
    
    def transform_sent(self, sent):
        
        sent = sent.split('\n')
        
        sent_id = int(sent[0][12:])
        text = sent[1][9:]
        tokens = [self.transform_token(token) for token in sent[2:]]
        
        return Sentence(sent_id, text, tokens)
    
    def transform_item(self, x):
        return [self.transform_sent(sent) for sent in x.strip().split('\n\n')]
        