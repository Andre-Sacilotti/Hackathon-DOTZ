# Hackathon-DOTZ

This hackathon gives an dataset with title of the product, and their respective category and sub-category, our work are to get the best classificator, and they will compare to the classifier they already uses

Public Leaderboard: Third Place

Final Leaderboard: *Waiting*

## Summary

This is an NLP problem with an extremly imbalanced dataset, so i made an ensemble of pre-trained DistilRoBERTa and many others machine learning algorithmns with TF-IDF

## Ensembles

For **Category** i stacked:

  • Multinomial Naive Bayes [Alpha=0.0025]

  • Complement Naive Bayes [Alpha = 1.0]

• MultiLayer Perceptron [optimizer = Adam, (500,0)
e (2700,540) dense layers, activation = relu,
output_activation = softmax]

• DistilRoberta [maxlen = 100, 20 epochs, batch_size =
128]

• KNN [n_neighbors=12,weights=distance]

• Random Forest [class_weight= balanced,
n_estimators=80]

And then for the **Sub-Category** i used:


• Multinomial Naive Bayes [Alpha = 0.01]

• Complement Naive Bayes [Alpha = 1.0]

• MultiLayer Perceptron [optimizer = Adam, (200,0)
e (2700,540) dense layers, activation = relu,
output_activation = softmax]

• DistilRoberta [maxlen = 100, 20 epochs, batch_size =
128]

• BERT-Base-Portuguese-Cased [maxlen = 100, 25
epochs, batch_size = 128]


## How to use

Before anything you need to download the models file, which has arround 2GB, in this <a href = ''>Link</a> (i will update soon)

Here you can find the ipython notebook where it can run each stage, but you can also execute the python script, following the commands

```
pip install -r requirements.txt
```
And then,

```
python main.py data/Hackathon_Base_Teste.csv
```
Changing the csv file for your file. It takes 0.03 seconds per data using an P-5000 GPU and arround 0.3 seconds per data using an I7-3770K. Moreover it uses at least 3GB RAM.




