import os
import cutlet
from tqdm import tqdm
katsu = cutlet.Cutlet()
katsu.use_foreign_spelling = False
new_data = []
for filename in os.listdir('../data'):
    if "translated" in filename:
        print("Romanizing ", filename, "...")
        with open('../data/' + filename, 'r', encoding="utf-8") as f:
            lines = f.readlines()
        for num, line in tqdm(enumerate(lines[1:])):
            idx, text, label, sem_type, language = line.strip().split("\t")
            if language == "ja":
                romanized = katsu.romaji(text)
            else:
                romanized = text
            new_data.append("\t".join([idx, romanized, label, sem_type, language]))
        with open('../data/' + filename, 'w', encoding="utf-8") as f:
            for line in new_data:
                f.write(line + "\n")