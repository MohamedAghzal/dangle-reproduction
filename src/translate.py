from googletrans import Translator
import os
import pandas as pd
import time

translator = Translator()

def translate_nl(file_name):
    languages = ['fr', 'es', 'ar', 'de', 'en', 'ja']

    df = pd.read_csv(
        os.path.join(os.path.dirname(__file__), file_name), 
        delimiter='\t', 
        names=["text", "label", "sem_type"]
    )

    translated_data = []

    for i in range(df.shape[0]):
        txt = df.iloc[i]['text']
        parse = df.iloc[i]['label']
        sem_type = df.iloc[i]['sem_type']

        for dest in languages:
            new = txt
            if(dest != 'en'):
                new = translator.translate(txt, src='en', dest=dest).text
            
            print(i, dest, new)
            translated_data.append([new, parse, sem_type, dest])
       
        if(i % 300 == 0 and i > 0):
            time.sleep(60)
    
    ret_df = pd.DataFrame(translated_data, columns=["text", "label", "sem_type", "language"])
    return ret_df


train_translated = translate_nl("../data/test.tsv")
train_translated.to_csv(os.path.dirname(__file__), "../data/test_translated.tsv", sep="\t")

