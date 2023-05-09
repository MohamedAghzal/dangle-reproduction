Here is the code which accompanies our NLP Project:

The src directory includes the code which we used to produce the T5 semantic parser as well as code for translating the data.
  - We attempted to reproduce the process of dangle but we were unable to get it working.
  - The Baseline.py and Baseline_Multilingual.py implement our standard T5 semantic parser and a multlingual variant.
  
The data directory contains the data on which our parsers were trained and evaluated.
  - The translated data sets are very large because they contain the translations of each sample in every language.
  - The "trimmed" translated data sets only contain one language per sample and an even distribution of languages.
