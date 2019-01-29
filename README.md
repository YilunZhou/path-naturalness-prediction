# path-naturalness-prediction

This is the repository for the code and data used in the paper *Predicting ConceptNet Path Quality Using Crowdsourced Assessments of Naturalness* by Yilun Zhou, Steven Schockaert, and Julie Shah, in proceedings of the The Web Conference 2019. For any questions, please contact Yilun Zhou at yilun@mit.edu. 

### Dataset
We provide three datasets, used in the three settings of Section 4.3. They are in the folders `data/science`, `data/money`, and `data/open-domain` respectively. Each folder contains two files: 
1. `paths.pkl` is the database of all sampled paths. 
2. `answers.txt` is the collected human ratings of pairwise comparison. 

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
# forward and reverse are both dictionaries and have the same structure. They have two fields, 'short' and 'text'. Short is a diagrammatic representation of the path. Text is a natural language representation of the path. Both are strings
>>> f_text = forward['text']
>>> f_short = forward['short']
>>> r_text = reverse['text']
>>> r_short = reverse['short']
>>> print f_text
# something like 'Card is a different concept from Paper. Paper is related to Part. Part is related to Hand. '
>>> print f_short
# something like 'Card <--DistinctFrom--> Paper <--RelatedTo--> Part <--RelatedTo--> Hand '
>>> print r_short
# something like 'Hand <--RelatedTo--> Part <--RelatedTo--> Paper <--DistinctFrom--> Card '
```
