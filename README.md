# path-naturalness-prediction

This is the repository for the code and data used in the paper *Predicting ConceptNet Path Quality Using Crowdsourced Assessments of Naturalness* by Yilun Zhou, Steven Schockaert, and Julie Shah, in proceedings of the The Web Conference 2019. For any questions, please contact Yilun Zhou at yilun@mit.edu. 

### Dataset
We provide three datasets, used in the three settings of Section 4.3. They are in the folders `data/science`, `data/money`, and `data/open-domain` respectively. For `science` and `money`, each folder contains two files: 
1. `paths.pkl` is the database of all sampled paths. 
2. `answers.txt` is the collected human answers of pairwise comparison. 

##### `paths.pkl`
This is a Python pickle file, which can be used in Python as 
```python
>>> import pickle
>>> paths = pickle.load(open('paths.pkl', 'rb')) 
# paths is dictionary, with keys being a string-represented numeric IDs, and values being problem contents
>>> keys = paths.keys()
>>> print keys[0] # something like '34672'
>>> path = paths[keys[0]]
# a path is a dictionary, with keys being 'forward' and 'reverse', representing the two directions
>>> forward = path['forward']
>>> reverse = path['reverse']
# forward and reverse are both dictionaries and have the same structure. 
# They have two fields, 'short' and 'text'. Short is a diagrammatic representation of the path. 
# Text is a natural language representation of the path. Both are strings. 
>>> f_text = forward['text']
>>> f_short = forward['short']
>>> r_text = reverse['text']
>>> r_short = reverse['short']
# a trailing whitespace is present in all these strings. 
>>> print f_text
# something like 'Card is a different concept from Paper. Paper is related to Part. Part is related to Hand. '
>>> print f_short
# something like 'Card <--DistinctFrom--> Paper <--RelatedTo--> Part <--RelatedTo--> Hand '
>>> print r_short
# something like 'Hand <--RelatedTo--> Part <--RelatedTo--> Paper <--DistinctFrom--> Card '
```
##### `answers.txt`
This is a text file, with each line being a pairwise comparison. Each line is in the form of `A_B_C`, where (`A`, `B`) is the pair provided to the participant, and `C` is the one chosen to be more natural (`C` is either `A` or `B`). The format of `A`, `B`, and `C` is the numeric ID followed by `f` or `r` which indicates the path direction. For example, `34672r` represents the path `Hand <--RelatedTo--> Part <--RelatedTo--> Paper <--DistinctFrom--> Card ` in the above example. 

For `open-domain`, paths of different lengths are in separate pickle files (the length of path is the number of vertices on the path), but file format is the same. In addition, the path ID in `answers.txt` are also appended with the length, since the original path IDs are not guaranteed to be unique across different lengths. 
