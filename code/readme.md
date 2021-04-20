## Repository for code of the paper:

Using text feature to improve valence prediction 
in dimensional speech emotion recognition

accepted in oCOCOSDA 2020

Experiments:
- text only
- acoustic only
- acoustic+text

Database:  
- IEMOCAP

Results (table/chart) for each database
1. Result of single-dimension valence prediction
2. Result multi-dimension speech emotion recognition --> see chart in the paper:   

| Modaliy    | V | A | D | Mean |    
| -----------|---|---|---|------|  
| Acoustic   |0.183|0.577|0.444 | 0.401 |   
| Acoustic+Text  |0.421 | 0.590 | 0.471 | 0.498 |  

3. Impact of using different pre-trained word vector (bert also?) --> Low.

We already observerd w2v and fasttext, but GloVe obtain better.
