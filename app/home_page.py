from fastapi import FastAPI
import os
import shutil
import ai_models
import models
import jsonlines
import datetime
from preprocess_data import final_preprocessing


users = {}
deliveries = {}
products = {}
sessions = {}
predictions = {}

app = FastAPI()

basic_classifier = ai_models.Classifier("models/basic_classifier.pkl")
target_classifier = ai_models.Classifier("models/target_classifier.pkl")


@app.get("/get-users")
def read_users():
    return users

@app.get("/get-sessions")
def read_sessions():
    return sessions


@app.get("/get-deliveries")
def read_deliveries():
    return deliveries


@app.get("/get-products")
def read_products():
    return products

@app.get("/get-predictions")
def read_predictions():
    return predictions

@app.get("/get-basic-prediction/{session_id}")
def get_basic_prediction(session_id: int):
    return {f"predictions for {session_id}": get_prediction(basic_classifier, session_id)}


@app.get("/get-target-prediction/{session_id}")
def get_target_prediction(session_id: int):
    return {f"predictions for {session_id}": get_prediction(target_classifier, session_id)}


@app.get("/get-both-models-prediction/{session_id}")
def get_both_prediction(session_id: int):
    basic_prediction = get_prediction(basic_classifier, session_id)
    target_prediction = get_prediction(target_classifier, session_id)
    both_predicitons = add_predictions(session_id, target_prediction, basic_prediction)
    return both_predicitons


@app.get("/save-all-data-and-predictions")
def get_all_predictions():
    predict_all()
    save_all_data()
    return {"Message": "All data and predictions saved"}


@app.post("/add-user")
def add_user(user: models.User):
    if user.user_id in users:
        return {"Error": "item already exist"}
    users[user.user_id] = {
        "user_id": user.user_id,
        "name": user.name,
        "city": user.city,
        "street": user.street
    }
    return users[user.user_id]


@app.post("/add-delivery")
def add_delivery(delivery: models.Delivery):
    if delivery.purchase_id in deliveries:
        return {"Error": "item already exist"}
    deliveries[delivery.purchase_id] = {
        "purchase_id": delivery.purchase_id,
        "purchase_timestamp": delivery.purchase_timestamp,
        "delivery_timestamp": delivery.delivery_timestamp,
        "delivery_company": delivery.delivery_company
        }
    return deliveries[delivery.purchase_id]


@app.post("/add-product")
def add_product(product: models.Product):
    if product.product_id in products:
        return {"Error": "item already exist"}
    products[product.product_id] = {
        "product_id": product.product_id,
        "product_name": product.product_name,
        "category_path": product.category_path,
        "price": product.price
        }
    return products[product.product_id]


@app.post("/add-session-row")
def add_session_one_row(session_row: models.Session):
    if int(session_row.session_id) not in sessions:
        sessions[int(session_row.session_id)] = []
    sessions[int(session_row.session_id)].append({
        "session_id": session_row.session_id,
        "timestamp": session_row.timestamp,
        "user_id": session_row.user_id,
        "product_id": session_row.product_id,
        "event_type": session_row.event_type,
        "offered_discount": session_row.offered_discount,
        "purchase_id": session_row.purchase_id,
        })
    return sessions[int(session_row.session_id)]


def predict_all():
    for session_id in sessions.keys():
        basic_prediction = get_prediction(basic_classifier, session_id)
        target_predition = get_prediction(target_classifier, session_id)
        prediction = add_predictions(session_id, target_predition, basic_prediction)


def add_predictions(session_id: int, target_prediction: int, basic_prediction: int):
    predictions[session_id] = {
        "session_id": session_id,
        "target_prediction": target_prediction,
        "basic_prediction": basic_prediction,
    }
    return predictions[session_id]


def save_all_data():
    for session_id in sessions.keys():
        save_data(sessions[session_id], "server_data/sessions.jsonl")
        save_data(users[sessions[session_id][-1]["user_id"]], "server_data/users.jsonl")
        save_data(products[sessions[session_id][-1]["product_id"]], "server_data/products.jsonl")
        save_data(deliveries[sessions[session_id][-1]["purchase_id"]], "server_data/deliveries.jsonl")
        save_data(predictions[session_id], "server_data/predictions.jsonl")


def clear_directory(path_to_directory):
    for filename in os.listdir(path_to_directory):
        file_path = os.path.join(path_to_directory, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)


def get_prediction(model, session_id):
    clear_directory('./temp_data')
    file_name = str(datetime.datetime.now())
    special_chars = " .:-"
    for special_char in special_chars:
        file_name = file_name.replace(special_char, '_')
    save_data(sessions[session_id], f"temp_data/{file_name}_sessions.jsonl")
    save_data(users[sessions[session_id][-1]["user_id"]], f"temp_data/{file_name}_users.jsonl")
    save_data(products[sessions[session_id][-1]["product_id"]], f"temp_data/{file_name}_products.jsonl")
    save_data(deliveries[sessions[session_id][-1]["purchase_id"]], f"temp_data/{file_name}_deliveries.jsonl")
    data_df = final_preprocessing(
        f"temp_data/{file_name}_users.jsonl",
        f"temp_data/{file_name}_products.jsonl",
        f"temp_data/{file_name}_deliveries.jsonl",
        f"temp_data/{file_name}_sessions.jsonl"
        )
    return model.predict(data_df).tolist()[0]


def save_data(data_line, filename: str):
    if type(data_line) == list:
        with jsonlines.open(filename, mode='a') as writer:
            for item in data_line:
                writer.write(item)
    else:
        with jsonlines.open(filename, mode='a') as writer:
            writer.write(data_line)