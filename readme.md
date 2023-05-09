Here is the code which accompanies our NLP Project:

The src directory includes the code which we used to produce the T5 semantic parser as well as code for translating the data.
  - We attempted to reproduce the process of dangle but we were unable to get it working.
  - The Baseline.py and Baseline_Multilingual.py implement our standard T5 semantic parser and a multlingual variant.
  - Below are some command lines to use this code:

Training the monolingual model:
  ```
  python Baseline.py train --cuda --train_dir ../data/train.tsv --val_dir ../data/dev.tsv --test_dir ../data/test.tsv --T5_modelname t5-base --save_dir ../models/baseline/ --epochs 200 --batch_size 64 --lr 0.0002
  ```
Evaluating the monolingual model:
  ```
  python Baseline.py evaluate --cuda --test_dir ../data/gen.tsv --checkpoint_dir ../models/baseline/ --batch_size 64
  ```
Training the multilingual model:
  ```
  python Baseline_Multilingual.py train --cuda --train_dir ../data/train_translated_trimmed.tsv --val_dir ../data/dev_translated_trimmed.tsv --test_dir ../data/test_translated_trimmed.tsv --T5_modelname google/mt5-small --save_dir ../models/baseline-translated/ --epochs 200 --batch_size 16 --lr 0.00002
  ```  
Evaluating the multilingual model:
  ```
  python Baseline_Multilingual.py evaluate --cuda --test_dir ../data/gen_translated.tsv --checkpoint_dir ../models/baseline-translated/ --batch_size 32
  ```

  
The data directory contains the data on which our parsers were trained and evaluated.
  - The translated data sets are very large because they contain the translations of each sample in every language.
  - The "trimmed" translated data sets only contain one language per sample and an even distribution of languages.
