import requests
from requests.models import Response
import jsonlines
import json


def read_from_file(filename):
    lines = []
    with jsonlines.open(filename) as reader:
        for obj in reader:
            lines.append(obj)
    return lines


def put_data(url, filename):
    data = read_from_file(filename)
    responses = []
    for item in data:
        responses.append(requests.post(url, data=json.dumps(item)))
    return responses


if __name__ == "__main__":
    dir_name = 'test_data/'
    session_id = 425
    
    users_response = put_data('http://127.0.0.1:8000/add-user', f'{dir_name}test_users.jsonl')
    deliveries_response = put_data('http://127.0.0.1:8000/add-delivery', f'{dir_name}test_deliveries.jsonl')
    products_response = put_data('http://127.0.0.1:8000/add-product', f'{dir_name}test_products.jsonl')
    sessions_responses = put_data('http://127.0.0.1:8000/add-session-row', f'{dir_name}/test_sessions.jsonl')

    print("Basic model prediction:")
    print(requests.get(f'http://127.0.0.1:8000/get-basic-prediction/{session_id}').content)

    print("Target model prediction:")
    print(requests.get(f'http://127.0.0.1:8000/get-target-prediction/{session_id}').content)

    print("All predictions:")
    requests.get('http://127.0.0.1:8000/save-all-data-and-predictions/')
    response_data = requests.get('http://127.0.0.1:8000/get-predictions/').json()
    for key in response_data.keys():
        print(response_data[key])