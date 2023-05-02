from googletrans import Translator
import os
import pandas as pd
import time

def translate_nl(file_name):
    print("Creating Translator...")
    translator = Translator()
    print("Translator Created.")

    languages = ['fr', 'es', 'ar', 'de', 'en', 'ja']
    
    print("Collecting data from: ", file_name, "...")
    df = pd.read_csv(
        os.path.join(file_name), 
        delimiter='\t', 
        names=["text", "label", "sem_type"]
    )
    print("Finished collecting data.")

    translated_data = []
    
    for i in range(df.shape[0]):
        txt = df.iloc[i]['text']
        parse = df.iloc[i]['label']
        sem_type = df.iloc[i]['sem_type']
        print("Translating: ", txt)
        for dest in languages:
            new = txt
            if(dest != 'en'):
                flag = True
                while flag:
                    try:
                        new = translator.translate(txt, src='en', dest=dest).text
                        flag = False
                    except Exception as e:
                        print(e)
                        print("Waiting a minute and trying again...")
                        time.sleep(60)
            print(i, dest, new)
            translated_data.append([new, parse, sem_type, dest])
       
        if(i % 300 == 0 and i > 0):
            time.sleep(60)
    
    ret_df = pd.DataFrame(translated_data, columns=["text", "label", "sem_type", "language"])
    return ret_df

if __name__ == "__main__":
    # train_translated = translate_nl("../data/test.tsv")
    # train_translated.to_csv("../data/test_translated.tsv", sep="\t")

    # train_translated = translate_nl("../data/dev.tsv")
    # train_translated.to_csv("../data/dev_translated.tsv", sep="\t")

    train_translated = translate_nl("../data/train.tsv")
    train_translated.to_csv("../data/train_translated.tsv", sep="\t")
