# TwitterSentiment
Simple tool for the Twitter sentiment analysis.
# Getting Started
Create Twitter account and apply for a developer account to access Twitter API.
Generate customer api keys, access tokens and update file `predict.py` with your keys and tokens.
Install all the requirements.
```bash
pip install -r requirements.txt
```
Run `predict.py` to train and predict.
```bash
python ./predict.py
```
When training is finished search can be made for the prediction, for example
```bash
Search: python
```
# Training Data
Training data used is downloaded from
http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip
More data for the training can be found from
http://help.sentiment140.com/home
