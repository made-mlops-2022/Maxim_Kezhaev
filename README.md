Кежаев Максим, Технопарк, ML-21
=====
hw1

Models to choose:
~~~
- LogisticRegression (or LR)
- KNeighborsClassifier (or KNN)
- GaussianNB (or GNB)
- DecisionTreeClassifier (or DT)
- SVC
- RandomForestClassifier (or RF)
~~~

How to run:
~~~
# Train
python train.py 

# Train with multirun
python train.py -m "++params.model_type=DT, KNN, RF"

# Predict with multirun
python predict.py -m "++params.model_type=DT, KNN, RF"

# Predict with included configs
python predict.py

# Predict with different path to save
python predict.py "save_paths.metrics= >put path<"
~~~
