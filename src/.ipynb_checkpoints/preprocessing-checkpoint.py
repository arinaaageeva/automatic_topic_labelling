import re
import time
import nltk
import math

from typing import List
from rusenttokenize import ru_sent_tokenize
from spacy.lang.ru import Russian
from spacy_russian_tokenizer import RussianTokenizer, MERGE_PATTERNS, SYNTAGRUS_RARE_CASES
from deeppavlov import configs, build_model
from pyaspeller import Word, YandexSpeller
from transliterate import translit
from rnnmorph.predictor import RNNMorphPredictor
from ufal.udpipe import Model, Pipeline, ProcessingError
from itertools import chain
from collections import Counter
from functools import reduce

from sklearn.base import BaseEstimator, TransformerMixin

from tqdm import tqdm

class Token:
    
    def __init__(self, \
                 token_id: int, \
                 form: str, \
                 lemma: str = None, \
                 upos: str = None, \
                 ner: str = None, \
                 feats: str = None, \
                 head: int = None, \
                 deprel: str = None, \
                 deps: str = None, \
                 space: bool = None):
        
        self.token_id = token_id
        self.form = form
        self.lemma = lemma
        self.upos = upos
        self.ner = ner
        self.feats = feats
        self.head = head
        self.deprel = deprel
        self.deps = deps
        self.space = space
        
    def copy(self):
        return Token(self.token_id, 
                     self.form, 
                     self.lemma, 
                     self.upos, 
                     self.ner, 
                     self.feats,
                     self.head,
                     self.deprel,
                     self.deps,
                     self.space)

class Sentence:
    
    def __init__(self, \
                 sent_id: int, \
                 text: str, \
                 tokens: List[Token] = None):
        
        self.sent_id = sent_id
        self.text = text
        self.tokens = tokens
        
    def copy(self):
        return Sentence(self.sent_id,
                        self.text,
                        None if self.tokens is None else [token.copy() for token in self.tokens])
        
class PreProcesser(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        
        print(type(self).__name__)
        time.sleep(1)
        
        return [self.transform_item(x) for x in tqdm(X)]
    
class Cleaner(PreProcesser):
    
    def transform_sent(self, sent):
        
        sent = sent.copy()
        sent.text = self.transform_text(sent.text)
        
        return sent
    
    def transform_item(self, x):
        
        if self.sent:
            return [self.transform_sent(sent) for sent in x]
        
        return self.transform_text(x)
    
class ReplaceChar(Cleaner):
    
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
    
class YandexSpeller(Cleaner):
    
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

#https://github.com/Mottl/ru_punkt
class RusSentTokenizer(PreProcesser):
    
    def transform_item(self, x):
        return [Sentence(sent_id, sent) for sent_id, sent in enumerate(nltk.sent_tokenize(x, language="russian"), 1)]
    
#https://github.com/aatimofeev/spacy_russian_tokenizer
class RusWordTokenizer(PreProcesser):
    
    def __init__(self):
        
        self.rus_word_tokenizer = Russian()
        
        pipe = RussianTokenizer(self.rus_word_tokenizer, MERGE_PATTERNS + SYNTAGRUS_RARE_CASES)
        self.rus_word_tokenizer.add_pipe(pipe, name='russian_tokenizer')
    
    def transform_text(self, text):
        return [Token(token_id, token.text) for token_id, token in enumerate(self.rus_word_tokenizer(text), 1)]
    
    def transform_sent(self, sent):
        
        sent = sent.copy()
        sent.tokens = self.transform_text(sent.text)
        
        return sent
    
    def transform_item(self, x):
        return [self.transform_sent(sent) for sent in x]
    
#http://docs.deeppavlov.ai/en/master/features/models/ner.html    
class RusWordTokenizer_NER(PreProcesser):
    
    def __init__(self):
        self.rus_ne_recognizer = build_model(configs.ner.ner_rus_bert)
        
    def transform_sent(self, sent, tokens, ners):
        
        sent = sent.copy()
        sent.tokens = [Token(token_id, token, ner=ner) for token_id, (token, ner) in enumerate(zip(tokens, ners), 1)]
        
        return sent
    
    def transform_item(self, x):
        return [self.transform_sent(sent, tokens, ners) for sent, tokens, ners in zip(*[x]+self.rus_ne_recognizer([sent.text for sent in x]))]
    
class SpaceDetecter(PreProcesser):
    
    def transform_sent(self, sent):
        
        sent = sent.copy()
        
        spaces = []
        text = sent.text
        
        for token in sent.tokens:
            
            index = text.find(token.form)
            spaces.append(bool(index))
            
            text = text[index+len(token.form):]
            
        spaces = spaces[1:]+[True]
        
        for token, space in zip(sent.tokens, spaces):
            token.space = space
        
        return sent
    
    def transform_item(self, x):
        return [self.transform_sent(sent) for sent in x]
    
class JoinByChar(PreProcesser):
    
    def __init__(self, chars_set=None):
        self.chars_set = {} if chars_set is None else chars_set
    
    def transform_sent(self, sent):
        
        sent = sent.copy()
        
        hyphen_indecies = [index for index, token in enumerate(sent.tokens) if token.form in self.chars_set]
        hyphen_indecies = [index for index in hyphen_indecies if 0 < index < len(sent.tokens)-1]
        hyphen_indecies = [index for index in hyphen_indecies if not(sent.tokens[index-1].space or sent.tokens[index].space)]
        
        match = lambda x: bool(re.match('\w', x))
        hyphen_indecies = [index for index in hyphen_indecies if match(sent.tokens[index-1].form) and match(sent.tokens[index+1].form)]
        
        while hyphen_indecies:
            
            index = hyphen_indecies[0]
            tokens = sent.tokens[index-1:index+2]
            
            token_id = tokens[0].token_id
            token_form = ''.join([token.form for token in tokens])
            token_ner = tokens[0].ner
            token_space = tokens[-1].space
            
            token = Token(token_id, token_form, ner=token_ner, space=token_space)
            sent.tokens = sent.tokens[:index-1]+[token]+sent.tokens[index+2:]
            
            hyphen_indecies = [index-2 for index in hyphen_indecies[1:]]
            
        for token_id, token in enumerate(sent.tokens, 1):
            token.token_id = token_id
        
        return sent
    
    def transform_item(self, x):
        return [self.transform_sent(sent) for sent in x] 
    
#https://github.com/IlyaGusev/rnnmorph
class MorphPredictor(PreProcesser):
    
    def __init__(self):
        self.rnnmorph = RNNMorphPredictor(language='ru')
    
    def translit(self, form):
        return (True, translit(form, 'ru')) if re.match(r'[a-zA-Z]+', form) else (False, form)
    
    def transform_sent(self, sent):
        
        sent = sent.copy()
        
        translit_flags, translit_forms = zip(*[self.translit(token.form) for token in sent.tokens])
        morth_forms = self.rnnmorph.predict(translit_forms)
        
        for token, morth_form, translit_flag in zip(sent.tokens, morth_forms, translit_flags):
            
            token.lemma = token.form.lower() if translit_flag else morth_form.normal_form
            token.upos = morth_form.pos
            token.feats = morth_form.tag
        
        return sent
        
    def transform_item(self, x):
        return [self.transform_sent(sent) for sent in x] 
    
#https://ufal.mff.cuni.cz/udpipe    
class SyntaxParser(PreProcesser):
    
    def __init__(self, model_path):
        
        self.parser_model = Model.load(model_path)
        self.parser_pipeline = Pipeline(self.parser_model, 'conllu', Pipeline.NONE, Pipeline.DEFAULT, 'conllu')
    
    def transform_item(self, x):
        return self.parser_pipeline.process(x, ProcessingError())
    
class Join_NER(PreProcesser):
    
    def join_form(self, tokens, use_lemma_form=False):
        
        spaces = [token.space for token in tokens]
        forms = [token.lemma if use_lemma_form else token.form for token in tokens]
        
        return  ''.join([form + (' ' if space else '') for form, space in zip(forms, spaces)]).strip()
    
    def join_ner(self, tokens):
        
        if len(tokens) == 1:
            return tokens[0]
        
        tokens_heads = [token.head for token in tokens]
        tokens_ids = [token.token_id for token in tokens]
       
        head = list(set(tokens_heads)-set(tokens_ids))[0]
        
        token = tokens[tokens_heads.index(head)].copy()
        token.form = self.join_form(tokens)
        token.lemma = self.join_form(tokens, use_lemma_form=True)
        token.ner = tokens[0].ner
        token.space = tokens[-1].space
        
        return token
    
    def tokens_ids_map(self, tokens, token):
        return {token_id:token.token_id for token_id in [token.token_id for token in tokens] if token_id != token.token_id}
    
    def append_token(self, tokens, ner_tokens, tokens_ids_map):
        
        if ner_tokens:
            
            token = self.join_ner(ner_tokens)
            tokens_ids_map.update(self.tokens_ids_map(ner_tokens, token))
            
            tokens.append(token)
            
        return tokens, tokens_ids_map
    
    def update_tokens_ids(self, sent, tokens_ids_map):
        
        for token in sent.tokens:
            if token.head in tokens_ids_map:
                token.head = tokens_ids_map[token.head]
                
        return sent
    
    def transform_sent(self, sent):
        
        sent = sent.copy()
        
        new_tokens = []
        ner_tokens = []
        
        tokens_ids_map = {}
        
        for token in sent.tokens:
            
            if token.ner[0] == 'B':
                
                new_tokens, tokens_ids_map = self.append_token(new_tokens, ner_tokens, tokens_ids_map)
                ner_tokens = [token]
                
            if token.ner[0] == 'I':
                
                if ner_tokens and (ner_tokens[0].ner[2:] == token.ner[2:]):
                    ner_tokens.append(token)
                    continue
                
                new_tokens, tokens_ids_map = self.append_token(new_tokens, ner_tokens, tokens_ids_map)
                ner_tokens = []
                
                token.ner = 'O'
                new_tokens.append(token)
                
            if token.ner[0] == 'O':
                
                new_tokens, tokens_ids_map = self.append_token(new_tokens, ner_tokens, tokens_ids_map)
                ner_tokens = []
                
                new_tokens.append(token)
             
        new_tokens, tokens_ids_map = self.append_token(new_tokens, ner_tokens, tokens_ids_map)
            
        sent.tokens = new_tokens
        sent = self.update_tokens_ids(sent, tokens_ids_map)
        
        tokens_ids_map = {}
        for new_token_id, token in enumerate(sent.tokens, 1):
            tokens_ids_map[token.token_id] = new_token_id
            token.token_id = new_token_id
        
        sent = self.update_tokens_ids(sent, tokens_ids_map)
        return sent
    
    def transform_item(self, x):
        return [self.transform_sent(sent) for sent in x]
    
class PhraseExtracter(PreProcesser):
    
    def __init__(self, min_count=1, max_len=float('inf')):
        
        self.min_count = min_count
        self.max_len = max_len
        
        self.vocabulary = Counter()
        self.total_tokens = 0
        
    def crop_vocabulary(self, vocabulary):
        return Counter(dict(filter(lambda x: x[1] >= self.min_count, vocabulary.items())))

    def init_boundaries(self, n_tokens):
        return list(zip(range(n_tokens), range(1, n_tokens+1)))
    
    def join(self, tokens):
        return ''.join([token.lemma + (' ' if token.space else '') for token in tokens]).strip()
    
    def fit(self, X, y=None):
        
        sent_tokens = [sent.tokens for sent in chain(*X)]
        sent_n_tokens = [len(tokens) for tokens in sent_tokens]
        
        self.max_len = min(max(sent_n_tokens), self.max_len)
        
        self.vocabulary = self.crop_vocabulary(Counter([token.lemma for token in chain(*sent_tokens)]))
        self.total_tokens = sum(self.vocabulary.values())
        
        print(f'Fit {type(self).__name__}')
        time.sleep(1)
        
        sent_boundaries = [self.init_boundaries(n_tokens) for n_tokens in sent_n_tokens]
        
        for _ in tqdm(range(self.max_len)):
            
            if not sent_boundaries:
                break
            
            phrases = []
            
            new_sent_tokens = []
            new_sent_n_tokens = []
            new_sent_boundaries = []
            
            for tokens, n_tokens, boundaries in zip(sent_tokens, sent_n_tokens, sent_boundaries):
        
                new_boundaries = []        
                for begin_index, end_index in boundaries:
                    
                    if (end_index != n_tokens) and \
                       (self.join(tokens[begin_index:end_index]) in self.vocabulary):
                        
                        end_index += 1
                        phrases.append(self.join(tokens[begin_index:end_index]))
                        new_boundaries.append((begin_index, end_index))
                        
                if new_boundaries:
                    new_sent_tokens.append(tokens)
                    new_sent_n_tokens.append(n_tokens)
                    new_sent_boundaries.append(new_boundaries)
            
            sent_tokens = new_sent_tokens
            sent_n_tokens = new_sent_n_tokens
            sent_boundaries = new_sent_boundaries
            
            self.vocabulary += self.crop_vocabulary(Counter(phrases))
        
        return self
    
    def sig(self, tokens, begin_index, separat_index, end_index):
        
        sig = -float('inf')
        
        phrase = self.join(tokens[begin_index:end_index])
        phrase_count = self.vocabulary[phrase]
        
        if phrase_count:
            
            begin_part = self.join(tokens[begin_index:separat_index])
            end_part = self.join(tokens[separat_index:end_index])
            
            begin_part_count = self.vocabulary[begin_part]
            end_part_count = self.vocabulary[end_part]
        
            sig = (begin_part_count*end_part_count)/self.total_tokens
            sig = (phrase_count - sig)/math.sqrt(phrase_count)
        
        return sig
    
    def transform_sent(self, sent):
        
        sent = sent.copy()
        
        phrases = []
        
        n_tokens = len(sent.tokens)
        boundaries = self.init_boundaries(n_tokens)
        
        for epoch in range(n_tokens-1):
            
            max_sig = -float('inf')
            max_boundary_index = -1
            
            for index in range(len(boundaries)-1):
                
                begin_index, separat_index = boundaries[index]
                _, end_index = boundaries[index+1]
                
                if (begin_index-end_index+1) > self.max_len:
                    continue
                    
                phrase_tokens = sent.tokens[begin_index:end_index]
                phrase_token_ids = [token.token_id for token in phrase_tokens]
                phrase_token_heads = [token.head for token in phrase_tokens]
                
                if len(set(phrase_token_heads)-set(phrase_token_ids)) > 1:
                    continue
            
                sig = self.sig(sent.tokens, begin_index, separat_index, end_index)
                
                if sig > max_sig:
                    max_sig = sig
                    max_boundary_index = index
                    
            if max_sig < 0: 
                break
                
            begin_boundary, _ = boundaries[max_boundary_index]
            _, end_boundary = boundaries.pop(max_boundary_index+1)
            
            boundaries[max_boundary_index] = (begin_boundary, end_boundary)
            phrases.append((begin_boundary, end_boundary, max_sig))
        
        return phrases
    
    def transform_item(self, x):
        return [self.transform_sent(sent) for sent in x]
    
class CoNLLUFormatEncoder(PreProcesser):
    
    def transform_token(self, token):
        
        encode_nan = lambda x: '_' if x is None else x
        encode_space = lambda x: 'SpaceAfter=No' if x is False else '_'
        
        conllu_token = f'{token.token_id}\t'
        conllu_token += f'{token.form}\t'
        conllu_token += f'{encode_nan(token.lemma)}\t'
        conllu_token += f'{encode_nan(token.upos)}\t'
        conllu_token += f'{encode_nan(token.ner)}\t'
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

        sent_id = sent[0][12:]
        text = sent[1][9:]
        tokens = [self.transform_token(token) for token in sent[2:]]

        return Sentence(sent_id, text, tokens)
    
    def transform_item(self, x):
        return [self.transform_sent(sent) for sent in x.strip().split('\n\n')]
    
class VowpalWabbitFormatEncoder(PreProcesser):
    
    def __init__(self, suffix='', upos_set=None, get_token_form=None, split_modalities=False, get_modality=None, rename_modalities=None):
        
        self.suffix = suffix
        self.upos_set = upos_set
        self.get_token_form = get_token_form
        self.split_modalities = split_modalities
        self.get_modality = get_modality
        self.rename_modalities = rename_modalities
        
    def transform_modality(self, modality):
        return ' '.join([token.replace(':', '') + ('' if count == 1 else f':{count}') for token, count in Counter(modality).items()])
    
    def transform_item(self, x):
        
        tokens = chain(*[sent.tokens for sent in x])
        
        if self.upos_set is not None:
            tokens = filter(lambda x: x.upos in self.upos_set, tokens)
        
        modalities = {}
        if self.split_modalities:
            for token in tokens:
                token_modality = self.get_modality(token)
                modalities.setdefault(token_modality, [])
                modalities[token_modality].append(self.get_token_form(token))
        
        else:
            modalities['tokens'] = [self.get_token_form(token) for token in tokens]
            
        if self.rename_modalities is not None:
            for old_name, new_name in self.rename_modalities.items():
                modalities[new_name] = modalities.pop(old_name) if old_name in modalities else []
        
        return ' '.join([f'|{self.suffix}{name} {self.transform_modality(modality)}' for name, modality in modalities.items()])