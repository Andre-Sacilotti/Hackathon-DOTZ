import sys

import ktrain
import numpy as np
import pandas as pd
from joblib import load
from ktrain import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

import preprocessing

if __name__ == "__main__":
    path = str(sys.argv[1])
    print(
        "Bem-vindo, estou processando seus dados bipbipbop(barulho de robo), aguarde um pouco, para 10000 dados pode demorar em media 5 minutos usando uma GPU ")

prep = preprocessing.preprocessing()

df = pd.read_csv('data/Hackathon_Base_Treino_comdep.csv')
df = prep.duplicate(df)
print(path)
try:
    print('AAAAAAAAAAAAAAA')
    dfsub = pd.read_csv(path)
except:
    print("Ei, parece que você colocou um diretorio errado para o arquivo CSV")
target_cat = prep.labelencoder(df, 'CATEGORIA')
target_catsub = prep.labelencoder(df, 'SUB-CATEGORIA')


## Spliting data for ktrain preprocessing for CATEGORIA
x_train, x_test, y_train, y_test = train_test_split(df['DESCRIÇÃO PARCEIRO'], target_cat, test_size=0.1,
                                                    stratify=target_cat, random_state=42)

t3 = text.Transformer('distilroberta-base', maxlen=100, classes=np.unique(target_cat))
train3 = t3.preprocess_train(list(x_train), y_train)
test3 = t3.preprocess_test(list(x_test), y_test)
model3 = t3.get_classifier()
learner3 = ktrain.get_learner(model3, train_data=train3, val_data=test3, batch_size=6)
learner3.model.load_weights('models/cat/weights-19cat.hdf5')
## End of CATEGORIA loading


## Spliting data for ktrain preprocessing for SUB- CATEGORIA
x_train, x_test, y_train, y_test = train_test_split(df['DESCRIÇÃO PARCEIRO'], target_catsub, test_size=0.1,
                                                    stratify=target_catsub, random_state=42)

t2 = text.Transformer('distilroberta-base', maxlen=100, classes=np.unique(target_catsub))
train2 = t2.preprocess_train(list(x_train), y_train)
test2 = t2.preprocess_test(list(x_test), y_test)
model2 = t2.get_classifier()
learner2 = ktrain.get_learner(model2, train_data=train2, val_data=test2, batch_size=6)
learner2.model.load_weights('models/subcat/weights-20.hdf5')

t = text.Transformer('neuralmind/bert-base-portuguese-cased', maxlen=100, classes=np.unique(target_catsub))
train = t.preprocess_train(list(x_train), y_train)
test = t.preprocess_test(list(x_test), y_test)
model = t.get_classifier()
learner = ktrain.get_learner(model, train_data=train, val_data=test, batch_size=6)
learner.model.load_weights('models/subcat/weights-21.hdf5')
## End of SUB-CATEGORIA loading


## lOADING OTHER MODELS
nbsub = load('models/subcat/nb.joblib')
mlpsub = load('models/subcat/mlp.joblib')
cnbsub = load('models/subcat/cnb.joblib')

nb = load('models/cat/nbcat.joblib' + '.compressed')
mlp = load('models/cat/mlpcat.joblib' + '.compressed')
knn = load('models/cat/knncat.joblib' + '.compressed')
cnb = load('models/cat/cnbcat.joblib' + '.compressed')
rf = load('models/cat/cnbcat.joblib' + '.compressed')

## END OF LOADING

vect = TfidfVectorizer(max_features=10000).fit(x_train)

## LOADING WEIGHTS FROM SUBCATEGORIA

model_weightssub = {'BERT': 0.8365236754871361,
                    'BERT2': 0.5430888111169458,
                    'CNB': 0.051903171320763775,
                    'MLP': 0.5671334042175022,
                    'MULTINOMIALNB': 0.0020652658573350998}

model_dictsub = dict(zip(['BERT', 'MULTINOMIALNB', "CNB", "BERT2", 'MLP'],
                         [learner, nbsub, cnbsub, learner2, mlpsub]))

## END


## LOADING WEIGHTS FROM CATEGORIA

model_weights = {'BERT2': 0.8691302469936113,
                 'CNB': 0.26971209717568634,
                 'KNN': 0.0011019891538130155,
                 'MLP': 0.17810369061951897,
                 'MULTINOMIALNB': 0.7467697788345558,
                 'RF': 0.6671639043827671}

model_dict = dict(zip(['MULTINOMIALNB', "BERT2", 'MLP', 'RF', 'KNN', 'CNB'],
                      [nb, learner3, mlp, rf, knn, cnb]))

## END


### Sub-categoria CLASSIFICATION

aux = dfsub['DESCRIÇÃO PARCEIRO']

## Bert1
predictor = ktrain.get_predictor(learner.model, preproc=t)
pred = predictor.predict(list(aux), return_proba=True)
pred = pred * model_weightssub['BERT']
## Bert2
predictor = ktrain.get_predictor(learner2.model, preproc=t2)
pred3 = predictor.predict(list(aux), return_proba=True)
pred3 = pred3 * model_weightssub['BERT2']

## Others models
pred2 = 0
for model_name, model in model_dictsub.items():
    print(model_name)
    if ((model_name != "BERT") & (model_name != "BERT2")):
        pred2 += model_dictsub[model_name].predict_proba(vect.transform(aux)) * model_weightssub[model_name]

## Average
pred6 = (pred2 + pred + pred3) / sum(model_weightssub.values())
dfsub['SUB-CATEGORIA'] = prep.labelencoder(np.argmax(pred6, axis=1), 'SUB-CATEGORIA', inverse=True)
### CATEGORIA CLASSIFICATION

## Bert2
predictor = ktrain.get_predictor(learner3.model, preproc=t3)
pred3 = predictor.predict(list(aux), return_proba=True)
pred3 = pred3 * model_weights['BERT2']

## Others
pred2 = 0
for model_name, model in model_dict.items():
    print(model_name)
    if ((model_name != "BERT") & (model_name != "BERT2")):
        pred2 += model_dict[model_name].predict_proba(vect.transform(aux)) * model_weights[model_name]

## Average
print("Terminei de classficar, o resultado sera salvo em /out/answer.csv")
pred6 = (pred2 + pred3) / sum(model_weights.values())
dfsub['CATEGORIA'] = prep.labelencoder(np.argmax(pred6, axis=1), 'CATEGORIA', inverse=True)
dfsub.to_csv('ANSWER.csv', index=False)
