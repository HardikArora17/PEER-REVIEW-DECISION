

import pickle
import os
import jsonlines
import nltk
import json
nltk.download('punkt',quiet=True)
from tqdm.autonotebook import tqdm


if(os.path.isdir('input_files')!=True):
    os.mkdir('input_files')

def create_data():
    data_path=r'raw_dataset/dataset/aspect_data'
    r_data={}
    counter=0

    phrase_data={}
    check_data={}

    with jsonlines.open(os.path.join(data_path,'review_with_aspect.jsonl')) as f:
        pbar=f.iter()
        
        for line in tqdm(pbar,desc='Loading Phrases from Data'):
            id1=line['id']
            if(id1.startswith('NIPS')):
                continue
            s=line['text']
            labels=line['labels']
            
            if (id1 not in phrase_data.keys()):
                phrase_data[id1]={}
                check_data[id1]={}
                
            aspect={}
            for k in labels:
                
                if(k[2].startswith('summary')):
                    continue
                    
                a=k[2].split('_')
                
                if(a[0]=='meaningful'):
                    a=(a[0]+'_'+a[1],a[2])
                else:
                    a=(a[0],a[1])

                h=s[k[0]:k[1]]
                
                
                if(a[0] in aspect):
                    
                    if(a[1] not in aspect[a[0]]):
                        aspect[a[0]][a[1]]=[]
                        aspect[a[0]][a[1]].append(h)
                        
                    else:
                        aspect[a[0]][a[1]].append(h)
                            
                else:
                    aspect[a[0]]={}
                    aspect[a[0]][a[1]]=[]
                    aspect[a[0]][a[1]].append(h)
                    
            
            phrase_data[id1][s]=aspect
            check_data[id1][s]=labels
            counter+=1

            
    data_path=r'raw_dataset/dataset/'
    decision_data={}

    for conf in os.listdir(data_path):
        if(conf=='aspect_data' or conf.startswith("NIPS")):
            continue
            
        for dire in (os.listdir(os.path.join(data_path,conf))):
            if(dire.endswith('_content')):
                continue
            
            if(dire.endswith('_paper')): 
                for paper in tqdm(os.listdir(os.path.join(data_path,conf,dire)),desc=conf+" DECISIONS "+": done"):

                    with open(os.path.join(data_path,conf,dire,paper)) as out:
                        file1=json.load(out)

                    decision=file1['decision']
                    decision_data[file1['id']]='accept' if ('Accept' in decision or 'Track' in decision) else 'reject'
                    
                    
    print()
    print("Total number of papers    :",len(decision_data))
    print("Number of Accepted Papers :",list(decision_data.values()).count('accept'))
    print("Number of Rejected Papers :",list(decision_data.values()).count('reject'))


    with open("input_files/paper_review_phrases.pickle",'wb') as out:
        pickle.dump(phrase_data,out)
        
    with open("input_files/paper_decision.pickle",'wb') as out:
        pickle.dump(decision_data,out)