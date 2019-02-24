from collections import namedtuple

Sent = namedtuple('Sent', 'id text tokens')
Token = namedtuple('Token', 'id token lemma pos tag head deprel space')

def space_detecter(text, tokens):
    
    spaces = []
    for token in tokens:

        index = text.find(token)
        text = text[index+len(token):]

        spaces.append(bool(index))

    spaces = spaces[1:]
    spaces.append(False)
        
    return spaces

def rnnmorph_encoder(sentences):
    
    def sentence_encoder(sent, tokens):
    
        spaces = space_detecter(sent, [token.word for token in tokens])
        tokens = [Token(token_id, token.word, token.normal_form, token.pos, token.tag, None, None, space)
                  for token_id, (token, space) in enumerate(zip(tokens, spaces))]
        
        return sent, tokens
    
    sentences = [sentence_encoder(sent, tokens) for sent, tokens in sentences]
    sentences = [Sent(sent_id, sent, tokens)  for sent_id, (sent, tokens) in enumerate(sentences)]
    
    return sentences
    
def conllu_decoder(sentences):
    
    def token_decoder(token):
        
        token_conllu = f'{token.id+1}\t'
        token_conllu += f'{token.token}\t'
        token_conllu += f'{token.lemma}\t'
        token_conllu += f'{token.pos}\t_\t'
        token_conllu += f'{token.tag}\t'
            
        head = '_' if token.head is None else token.head+1
        deprel = '_' if token.deprel is None else token.deprel
        space = '_' if token.space else 'SpaceAfter=No'
            
        token_conllu += f'{head}\t{deprel}\t_\t{space}'
        
        return token_conllu
    
    sentences = [(sent.id+1, sent.text, '\n'.join([token_decoder(token) for token in sent.tokens])) for sent in sentences]
    sentences = [f'# sent_id = {sent_id}\n# text = {text}\n{tokens}' for sent_id, text, tokens in sentences]
    
    return '\n\n'.join(sentences)+'\n\n'

def conllu_encoder(sentences):
    
    def token_encoder(token):
        
        token = token.split('\t')
        
        head = None if token[6] == '_' else int(token[6])-1
        deprel = None if token[7] == '_' else token[7]
        space = True if token[9] == '_' else False
        
        return Token(int(token[0])-1, token[1], token[2], token[3], token[5], head, deprel, space)
    
    sentences = sentences.split('\n\n')[:-1]
    sentences = [sent.split('\n') for sent in sentences]
    sentences = [Sent(int(sent[0].replace('# sent_id = ', ''))-1, 
                      sent[1].replace('# text = ', ''), 
                      [token_encoder(token) for token in sent[2:]]) 
                 for sent in sentences]
    
    return sentences