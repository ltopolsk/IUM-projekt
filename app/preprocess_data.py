from ai_models import CategoryOnehot, CitiesVoc, StandardScaler
import pandas as pd
import numpy as np



def format_timestamps(dataframe, col_names):
    for name in col_names:
        dataframe[name] = pd.to_datetime(dataframe[name], format='%Y-%m-%dT%H:%M:%S')


def format_deliveries_df(dataframe):
    format_timestamps(dataframe, ["purchase_timestamp", "delivery_timestamp"])
    dataframe["deliver_time"] = (dataframe["delivery_timestamp"] - dataframe["purchase_timestamp"])
    dataframe["deliver_time"] = dataframe["deliver_time"] / np.timedelta64(1, 'D')


def add_month(dataframe, timestamp_name):
    dataframe["month"] = dataframe[timestamp_name].apply(lambda x: "%d" % (x.month))


def group_by_dataframe(dataframe, group_attr) -> pd.DataFrame:
    aggregation_functions = {
        'month': 'first', 
        'user_id': 'first',
        'offered_discount': 'last',
        'purchase_id': 'last',
        'event_type': 'count',
        'product_id': 'last',
    }
    grouped_dataframe = dataframe.groupby(dataframe[group_attr]).aggregate(aggregation_functions)
    return grouped_dataframe


def merge_dataframes(dataframe_left, dataframe_right, how='left') -> pd.DataFrame:
    return pd.merge(dataframe_left, dataframe_right, how=how)


def drop_columns(dataframe, col_names):
    dataframe.drop(col_names, axis=1, inplace=True)


def final_preprocessing(users_filename, products_filename, deliveries_filename, sessions_filename):
    sessions_df = pd.read_json(sessions_filename, lines=True)
    users_df = pd.read_json(users_filename, lines=True)
    deliveries_df = pd.read_json(deliveries_filename, lines=True)
    products_df = pd.read_json(products_filename, lines=True)
    format_deliveries_df(deliveries_df)
    add_month(sessions_df, "timestamp")
    sessions_grouped_df = group_by_dataframe(sessions_df, "session_id")
    sessions_user_grouped_df = merge_dataframes(sessions_grouped_df, users_df)
    sessions_user_deliver_grouped_df = merge_dataframes(sessions_user_grouped_df, deliveries_df)
    all_df = merge_dataframes(sessions_user_deliver_grouped_df, products_df)
    drop_columns(
        all_df,
        [
            "street",
            "purchase_timestamp",
            "delivery_timestamp",
            "purchase_id",
            "name",
            "product_name"
        ]
    )
    cities_voc = CitiesVoc()
    all_df["city"] = all_df.city.map(cities_voc.get_dictonary())
    one_hot_encoder = CategoryOnehot()
    encoded_data = one_hot_encoder.encode(all_df, ["category_path"])
    string_values = pd.DataFrame(
        encoded_data,
        columns=list(one_hot_encoder.encoder.categories_[0])
        )
    drop_columns(all_df, ["category_path"])
    all_df = pd.concat([string_values, all_df], axis=1)
    data_scaler = StandardScaler()
    scaled_data = data_scaler.transform_data(all_df.values)
    data_scaled_df = pd.DataFrame(scaled_data, index=all_df.index, columns=all_df.columns)

    return data_scaled_df.values
