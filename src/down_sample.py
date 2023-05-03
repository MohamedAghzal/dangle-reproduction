import pandas as pd
from tqdm import tqdm
languages = ['fr', 'es', 'ar', 'de', 'en', 'ja']

def new_data(filename):
    df = pd.read_csv(filename, delimiter="\t")
    data = []
    lang_id = 0
    for i in tqdm(range(df.shape[0])):
        if i%len(languages)+1 == len(languages):
            lang_id += 1
        if lang_id == len(languages):
            lang_id = 0
        if df.iloc[i].language == languages[lang_id]:
            data.append([df.iloc[i].text,
                        df.iloc[i].label,
                        df.iloc[i].sem_type,
                        df.iloc[i].language])
    return pd.DataFrame(data, columns=["text", "label", "sem_type", "language"])

if __name__ == "__main__":
    new_data("../data/dev_translated.tsv").to_csv("../data/dev_translated_trimmed.tsv", sep="\t")
    new_data("../data/test_translated.tsv").to_csv("../data/test_translated_trimmed.tsv", sep="\t")
    new_data("../data/train_translated.tsv").to_csv("../data/train_translated_trimmed.tsv", sep="\t")