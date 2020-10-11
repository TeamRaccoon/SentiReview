import re, os
import nltk
from nltk.tokenize import word_tokenize

def remove_stopword(stoplist_dir, txt):
    stoplist = []
    with open(os.path.join(stoplist_dir,'Stopwords.txt'),'r',encoding='utf-8') as sw: 
        for word in sw:
            stoplist.append(word.rstrip())
    
    word_tokens = word_tokenize(txt)
    result = []
    for w in word_tokens:
        if w not in stoplist:
                result.append(w)
                 
    new_review =' '.join(result)
    return new_review

#fine_tuning
def clean_text(stoplist_dir, txt):

    # remove special symbols
    txt =re.sub('[^\w\s]', ' ', txt)
    # remove blanks
    txt = ' '.join(txt.split())
    # make alphabet lower
    txt = txt.lower()
    #remove stopword
    txt = remove_stopword(stoplist_dir, txt)
    
    return txt 


