from statistics import mode
import pandas as pd
import numpy as np
import pickle
import os

from tensorflow import keras
from typing import List
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel

class Item(BaseModel):
    fr: float
    fc: float
    spo2: float
    tam: float
    condition: int

class ClinicalData(BaseModel):
    data: List[Item]

# Creación de una aplicación FastAPI:
app = FastAPI()
dirname = os.path.dirname(__file__)

@app.get("/")
async def root():
    return "It's running..."

@app.post("/predict")
async def predict(group: int, data: ClinicalData):
    items = []
    columns = ["SpO2", "FC", "FR", "TAM", "Condition"]
    for i in data.data:
        arr = [i.spo2, i.fc, i.fr, i.tam, i.condition]
        items.append(arr)

    df = pd.DataFrame(items, columns=columns)
    df = df.tail(70)

    dtc = calculateDecisionTree(df, group)
    rfc = calculateRandomForest(df, group)
    lrc = calculateLogisticRegression(df, group)
    predictedLstm = predictValuesLstm(df, group)
    predictedGru = predictValuesGru(df, group)

    return formatResult(predictedLstm, predictedGru, dtc, rfc, lrc)

def predictValuesGru(df, group):
    if group == 1:
        model = keras.models.load_model(os.path.join(dirname, "Models", "lstm_model_group_1.h5"), compile=False)
    elif group == 2:
        model = keras.models.load_model(os.path.join(dirname, "Models", "lstm_model_group_2.h5"), compile=False)
    elif group == 3:
        model = keras.models.load_model(os.path.join(dirname, "Models", "lstm_model_group_3.h5"), compile=False)
    else:
        model = keras.models.load_model(os.path.join(dirname, "Models", "lstm_model_group_4.h5"), compile=False)

    x, _ = create_test_data(df, 1)
    predicted = model.predict(x)
    predicted = (predicted.round()).tolist()

    return predicted

def predictValuesLstm(df, group):
    if group == 1:
        model = keras.models.load_model(os.path.join(dirname, "Models", "gru_model_group_1.h5"), compile=False)
    elif group == 2:
        model = keras.models.load_model(os.path.join(dirname, "Models", "gru_model_group_2.h5"), compile=False)
    elif group == 3:
        model = keras.models.load_model(os.path.join(dirname, "Models", "gru_model_group_3.h5"), compile=False)
    else:
        model = keras.models.load_model(os.path.join(dirname, "Models", "gru_model_group_4.h5"), compile=False)

    x, _ = create_test_data(df, 1)
    predicted = model.predict(x)
    predicted = (predicted.round()).tolist()

    return predicted

def calculateDecisionTree(df, group):
    if group == 1:
        model_c = pickle.load(open(os.path.join(dirname, "Models", "decision_tree_model_1_c.sav"), "rb"))
        model_s = pickle.load(open(os.path.join(dirname, "Models", "decision_tree_model_1_s.sav"), "rb"))
    elif group == 2:
        model_c = pickle.load(open(os.path.join(dirname, "Models", "decision_tree_model_2_c.sav"), "rb"))
        model_s = pickle.load(open(os.path.join(dirname, "Models", "decision_tree_model_2_s.sav"), "rb"))
    elif group == 3:
        model_c = pickle.load(open(os.path.join(dirname, "Models", "decision_tree_model_3_c.sav"), "rb"))
        model_s = pickle.load(open(os.path.join(dirname, "Models", "decision_tree_model_3_s.sav"), "rb"))
    else:
        model_c = pickle.load(open(os.path.join(dirname, "Models", "decision_tree_model_4_c.sav"), "rb"))
        model_s = pickle.load(open(os.path.join(dirname, "Models", "decision_tree_model_4_s.sav"), "rb"))

    sub_df = df.tail(5)
    sub_df.drop('Condition', inplace=True, axis=1)
    y_pred_1_c = model_c.predict(sub_df)
    modeInResult = mode(y_pred_1_c)

    if modeInResult == 1:
        y_pred_1_s = model_s.predict(sub_df)

    index = 0
    result = []
    for item in y_pred_1_c:
        if item == 1 and len(y_pred_1_s) > 0:
            result.append(y_pred_1_s[index])
        else:
            result.append(y_pred_1_c[index])
        index = index + 1

    return result

def calculateRandomForest(df, group):
    if group == 1:
        model_c = pickle.load(open(os.path.join(dirname, "Models", "rfc_model_1_c.sav"), "rb"))
        model_s = pickle.load(open(os.path.join(dirname, "Models", "rfc_model_1_s.sav"), "rb"))
    elif group == 2:
        model_c = pickle.load(open(os.path.join(dirname, "Models", "rfc_model_2_c.sav"), "rb"))
        model_s = pickle.load(open(os.path.join(dirname, "Models", "rfc_model_2_s.sav"), "rb"))
    elif group == 3:
        model_c = pickle.load(open(os.path.join(dirname, "Models", "rfc_model_3_c.sav"), "rb"))
        model_s = pickle.load(open(os.path.join(dirname, "Models", "rfc_model_3_s.sav"), "rb"))
    else:
        model_c = pickle.load(open(os.path.join(dirname, "Models", "rfc_model_4_c.sav"), "rb"))
        model_s = pickle.load(open(os.path.join(dirname, "Models", "rfc_model_4_s.sav"), "rb"))

    sub_df = df.tail(5)
    sub_df.drop('Condition', inplace=True, axis=1)
    y_pred_1_c = model_c.predict(sub_df)
    modeInResult = mode(y_pred_1_c)

    if modeInResult == 1:
        y_pred_1_s = model_s.predict(sub_df)

    index = 0
    result = []
    for item in y_pred_1_c:
        if item == 1 and len(y_pred_1_s) > 0:
            result.append(y_pred_1_s[index])
        else:
            result.append(y_pred_1_c[index])
        index = index + 1

    return result

def calculateLogisticRegression(df, group):
    if group == 1:
        model_c = pickle.load(open(os.path.join(dirname, "Models", "lrc_model_1_c.sav"), "rb"))
        model_s = pickle.load(open(os.path.join(dirname, "Models", "lrc_model_1_s.sav"), "rb"))
    elif group == 2:
        model_c = pickle.load(open(os.path.join(dirname, "Models", "lrc_model_2_c.sav"), "rb"))
        model_s = pickle.load(open(os.path.join(dirname, "Models", "lrc_model_2_s.sav"), "rb"))
    elif group == 3:
        model_c = pickle.load(open(os.path.join(dirname, "Models", "lrc_model_3_c.sav"), "rb"))
        model_s = pickle.load(open(os.path.join(dirname, "Models", "lrc_model_3_s.sav"), "rb"))
    else:
        model_c = pickle.load(open(os.path.join(dirname, "Models", "lrc_model_4_c.sav"), "rb"))
        model_s = pickle.load(open(os.path.join(dirname, "Models", "lrc_model_4_s.sav"), "rb"))

    sub_df = df.tail(5)
    sub_df.drop('Condition', inplace=True, axis=1)
    y_pred_1_c = model_c.predict(sub_df)
    modeInResult = mode(y_pred_1_c)

    y_pred_1_s = []
    if modeInResult == 1:
        y_pred_1_s = model_s.predict(sub_df)

    index = 0
    result = []
    for item in y_pred_1_c:
        if item == 1 and len(y_pred_1_s) > 0:
            result.append(y_pred_1_s[index])
        else:
            result.append(y_pred_1_c[index])
        index = index + 1

    return result

def formatResult(predictedLstm, predictedGru, dtc, rfc, lrc):
    minute05Lstm = None
    minute10Lstm = None
    minute20Lstm = None
    minute30Lstm = None
    minute45Lstm = None
    minute60Lstm = None

    minute05Gru = None
    minute10Gru = None
    minute20Gru = None
    minute30Gru = None
    minute45Gru = None
    minute60Gru = None

    if len(predictedLstm) > 5:
        minute05Lstm = predictedLstm[5][0]
    if len(predictedLstm) > 10:
        minute10Lstm = predictedLstm[10][0]
    if len(predictedLstm) > 20:
        minute20Lstm = predictedLstm[20][0]
    if len(predictedLstm) > 30:
        minute30Lstm = predictedLstm[30][0]
    if len(predictedLstm) > 45:
        minute45Lstm = predictedLstm[45][0]
    if len(predictedLstm) > 60:
        minute60Lstm = predictedLstm[60][0]

    if len(predictedGru) > 5:
        minute05Gru = predictedGru[5][0]
    if len(predictedGru) > 10:
        minute10Gru = predictedGru[10][0]
    if len(predictedGru) > 20:
        minute20Gru = predictedGru[20][0]
    if len(predictedGru) > 30:
        minute30Gru = predictedGru[30][0]
    if len(predictedGru) > 45:
        minute45Gru = predictedGru[45][0]
    if len(predictedGru) > 60:
        minute60Gru = predictedGru[60][0]
    
    

    return {
            "lstm": {
                "minute05": minute05Lstm,
                "minute10": minute10Lstm,
                "minute20": minute20Lstm,
                "minute30": minute30Lstm,
                "minute45": minute45Lstm,
                "minute60": minute60Lstm,
            },
            "gru": {
                "minute05": minute05Gru,
                "minute10": minute10Gru,
                "minute20": minute20Gru,
                "minute30": minute30Gru,
                "minute45": minute45Gru,
                "minute60": minute60Gru,
            },
            "dtc": {
                "minuteMinus4": int(dtc[0]),
                "minuteMinus3": int(dtc[1]),
                "minuteMinus2": int(dtc[2]),
                "minuteMinus1": int(dtc[3]),
                "minuteMinus0": int(dtc[4]),
            },
            "rfc": {
                "minuteMinus4": int(rfc[0]),
                "minuteMinus3": int(rfc[1]),
                "minuteMinus2": int(rfc[2]),
                "minuteMinus1": int(rfc[3]),
                "minuteMinus0": int(rfc[4]),
            },
            "lrc": {
                "minuteMinus4": int(lrc[0]),
                "minuteMinus3": int(lrc[1]),
                "minuteMinus2": int(lrc[2]),
                "minuteMinus1": int(lrc[3]),
                "minuteMinus0": int(lrc[4]),
            }
        }

def create_test_data(df, look_back):
    dataX1, dataX2 = [], []
    for i in range(len(df)-look_back-1):
        dataX1.append(df.iloc[i : i + look_back, 0:4].values)
        dataX2.append(df.iloc[i : i + look_back, 4].values)
        
    return np.array(dataX1), np.array(dataX2)

# def calculateKNeighbors(df, group):
#     if group == 1:
#         model = pickle.load(open(os.path.join(dirname, "Models", "knc_model_1.sav"), "rb"))
#     elif group == 2:
#         model = pickle.load(open(os.path.join(dirname, "Models", "knc_model_2.sav"), "rb"))
#     elif group == 3:
#         model = pickle.load(open(os.path.join(dirname, "Models", "knc_model_3.sav"), "rb"))
#     else:
#         model = pickle.load(open(os.path.join(dirname, "Models", "knc_model_4.sav"), "rb"))

#     sub_df = df.tail(5)
#     sub_df.drop('Condition', inplace=True, axis=1)
#     y_pred_1 = model.predict(sub_df)

#     av = 0
#     for i in y_pred_1:
#         av += y_pred_1[i]

#     return round(av/len(sub_df))

# def calculateMultilayerPerceptron(df, group):
#     if group == 1:
#         model = keras.models.load_model(os.path.join(dirname, "Models", "multilayer_perceptron_1.h5"), compile=False)
#     elif group == 2:
#         model = keras.models.load_model(os.path.join(dirname, "Models", "multilayer_perceptron_2.h5"), compile=False)
#     elif group == 3:
#         model = keras.models.load_model(os.path.join(dirname, "Models", "multilayer_perceptron_3.h5"), compile=False)
#     else:
#         model = keras.models.load_model(os.path.join(dirname, "Models", "multilayer_perceptron_4.h5"), compile=False)

#     sub_df = df.tail(5)
#     sub_df.drop('Condition', inplace=True, axis=1)
#     y_pred_1 = model.predict(sub_df)

#     av: float = 0
#     for value in y_pred_1:
#         av += abs(value[0].round())

#     return round(av/len(sub_df))


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="UCIP",
        version="1.0.0",
        description="Developed by Diego Andrés BAquero",
        routes=app.routes,
    )

    openapi_schema["paths"]["/api/auth"] = {
        "post": {
            "requestBody": {"content": {"application/json": {}}, "required": True}, "tags": ["Auth"]
        }
    }

    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi
