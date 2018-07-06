# Webservice für Praktikum "Grundlagen Maschinelles Lernen"

## Installation
Klonen des Repositorys.
```
git clone https://github.com/karnehm/simple_keras_rest.git
```
Trainiertes Model downloaden von `https://drive.google.com/open?id=19effzgC-4SkarjNYIAyDUKFio0NJTb5l`

Installation der notwendigen Software.
```
pip install numpy glob tensorflow keras flask
```
## Run server

Um die Software zu verwenden starten sie den Server mittels folgendem Befehl:

```
python run_keras_server.py 
```

## Verwendung

Um Predictions zu erhalten rufen sie den Webservice auf. Beispielhaft kann dies mittels dem Graphen durchgeführt werden, welcher sich im Repository befindet. So erhalten sie eine Prediction als Antwort.

```
curl -X POST -F image=@graph.png 'http://localhost:5000/predict'
```




