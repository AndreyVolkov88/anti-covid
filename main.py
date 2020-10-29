import os
import sys
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import wandb
from wandb.keras import WandbCallback
sys.path.append(os.path.join(os.path.dirname(__file__)))
from model import create_model
from preprocessing import process_data

wandb.init(project='covid_19')

data = process_data(r'E:\covid-19\LIDC data\LIDC-IDRI', 128, 64)
print(np.shape(data))
model = create_model((64, 128, 128, 1))
print(model.summary())
enc = OneHotEncoder()
y = enc.fit_transform(np.array([0, 1, 1, 0, 1, 0, 1, 0, 0, 1]).reshape(-1, 1)).toarray()
model.fit(data, y, epochs=3, callbacks=[WandbCallback()])
input()
