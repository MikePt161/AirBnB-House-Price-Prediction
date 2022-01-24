import pandas as pd
import numpy as np
from unidecode import unidecode
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from joblib import dump, load

def select_features(df_list):
    # Select features
    df_list = df_list[[
        ### LISTINGS ID - FOR USE IN SENTIMENT ANALYSIS
        "id",

        ### INFO ABOUT LOCATION & HOST
        "neighbourhood_cleansed",  # names of neighbourhoods
        # "host_since",  # older hosts = more credibility? (will not work on new host)
        "host_response_rate",  # linked to credibility - problematic for new entries
        "host_acceptance_rate",  # same
        "host_is_superhost",  # same (new host always false - how is superhost defined?)
        "host_listings_count",  # more listings = more experience = higher ratings?
        "host_identity_verified",  # identity verified/not higher ratings (is likely string)
        "latitude",  # LAT - LONG need preprocessing
        "longitude",

        ### INFO ABOUT ROOM
        "room_type",  # more meaningful & diverse than property type (check visualisation)
        "bathrooms_text",  # needs preprocessing -> int
        "bedrooms",  # !!!! Bedrooms and beds might contain the same amount of info
        "beds",  # run a correlation with bedrooms
        "accommodates",  # similar to beds, bedrooms
        "amenities",  # extract features and preprocess with T/F & 1-hot-encoding

        ### INFO ABOUT PRICE-RATINGS
        "number_of_reviews",
        # more reviews -> more likely to get booked -> more high reviews (does not apply to new clients)
        "reviews_per_month",  # Same

        ### RESPONSE
        "price",  # turn to int ? check missingness!!

    ]].copy()
    return df_list


def fix_rates(df_list):
    ##### Transform percentages represented as strings to float
    def rate_text_to_float(val):
        return float(val[:-1]) / 100 if pd.notna(val) else val

    df_list.loc[:, "host_response_rate"] = df_list["host_response_rate"].apply(rate_text_to_float)
    df_list.loc[:, "host_acceptance_rate"] = df_list["host_acceptance_rate"].apply(rate_text_to_float)

    return df_list


def fix_bools(df_list):
    ##### Transform text 't' / 'f' values to corresponding bool
    def string_to_bool(val):
        return int(val == 't')

    df_list.loc[:, "host_is_superhost"] = df_list["host_is_superhost"].apply(string_to_bool)
    df_list.loc[:, "host_identity_verified"] = df_list["host_identity_verified"].apply(string_to_bool)

    return df_list


def fix_neighbourhoods(df_list):
    df_list.loc[:, "neighbourhood_cleansed"] = df_list["neighbourhood_cleansed"].apply(
        lambda x: "neighbourhood_" + unidecode(x).lower().replace(" ", "_").replace("-", "_"))

    return df_list


def fix_amenities(df_list):
    def list_from_string(s):
        return [el.strip(' ').strip('"').replace(" ", "_") for el in s[1:-1].split(',')]

    df_list.loc[:, "amenities"] = df_list["amenities"].apply(list_from_string)

    return df_list


def create_bathrooms_number(df_list):
    """
    Input: dataframe with bathrooms column as text
    Output: dataframe with new "bathrooms_number" column (float) and dropped "bathrooms_text" column
    """

    df_list["bathrooms_number"] = df_list.bathrooms_text.str.split(pat=" ", expand=True)[0]
    df_list["bathrooms_number"].replace({"Half-bath": "0.5", "Shared": "1"}, inplace=True)
    df_list["bathrooms_number"] = df_list["bathrooms_number"].astype('float64')
    df_list.drop(columns=["bathrooms_text"], inplace=True, axis=1)
    return df_list


def fix_price(df_list):
    df_list.loc[:, "price"] = df_list["price"].apply(lambda x: float(x[1:].replace(',', '')) if pd.notna(x) else x)
    return df_list


def combine_pre_split_fixes(df_list):
    df_list = select_features(df_list)
    df_list = fix_rates(df_list)
    df_list = fix_bools(df_list)
    df_list = fix_neighbourhoods(df_list)
    df_list = fix_amenities(df_list)
    df_list = create_bathrooms_number(df_list)
    df_list = fix_price(df_list)

    return df_list


def split_dataset(df_list):
    X = df_list.drop(columns="price")
    y = df_list["price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


def find_bound(y_train, upper=True):
    iqr = np.subtract(*np.percentile(y_train, [75, 25]))

    if upper:

        upperbound = np.percentile(y_train, 75) + 1.5 * iqr

        return upperbound

    else:

        lowerbound = np.percentile(y_train, 25) - 1.5 * iqr

        return lowerbound


def remove_outliers(X_train, y_train):
    upper_bound = find_bound(y_train)
    X_train = X_train.loc[(y_train < upper_bound), :]
    y_train = y_train[(y_train < upper_bound)]
    return X_train, y_train


def save_fit_scaler(X_train):
    features_to_be_scaled = ['host_response_rate',
                            'host_acceptance_rate',
                            'host_listings_count',
                            'latitude',
                            'longitude',
                            'bedrooms',
                            'beds',
                            'accommodates',
                            'number_of_reviews',
                            'reviews_per_month'
                            ]

    scaler = preprocessing.StandardScaler()

    # Scaler will be fit on training data
    scaler = scaler.fit(X_train[features_to_be_scaled])

    dump(scaler, 'data/scaler.joblib')


def scale_features(data):
    """
    Scales specified features.

    ### Inputs:

    X_train: dataframe upon which the scaler will be fit
    X_test: specified test dataframe that will be transformed
    test: whether the scaler involves training or testing process

    #### Outputs:

    X_train or X_test: scaled dataframes
    """

    features_to_be_scaled = ['host_response_rate',
                             'host_acceptance_rate',
                             'host_listings_count',
                             'latitude',
                             'longitude',
                             'bedrooms',
                             'beds',
                             'accommodates',
                             'number_of_reviews',
                             'reviews_per_month'
                             ]

    scaler = load('data/scaler.joblib')

    data[features_to_be_scaled] = scaler.transform(data[features_to_be_scaled])

    return data

def generate_fill_values(X_train):
    mode_feats = ["neighbourhood_cleansed", "host_is_superhost", "host_identity_verified", "room_type"]

    fill_values_df = {}
    for feat in X_train.columns[[col not in ['id', 'amenities'] for col in X_train.columns]]:
        fill_values_df[feat] = X_train[feat].mode() if feat in mode_feats else X_train[feat].median()

    fill_values_df['amenities'] = '[]'

    fill_values_df = pd.DataFrame(fill_values_df)
    fill_values_df.to_csv("data/fill_values.csv", index=False)


def impute_missing_values(data):
    fill_values = pd.read_csv("data/fill_values.csv")
    fill_values_dict = fill_values.to_dict(orient='records')[0]
    return data.fillna(fill_values_dict)


def one_hot_amenities(data):
    """
    Performs one-hot encoding to fixed amenities.
    The most provided (>4000) amenities are used as one-hot features.

    ### Inputs

    df_list: dataframe

    ### Outputs

    df_list: dataframe with one-hot encodings on most frequent amenities.

    """

    amenities_cat = pd.read_csv('data/amenities_cat.csv')
    wanted_keys = amenities_cat.iloc[:, 0].tolist()
    #### Initialise list
    one_hot = []

    for j in wanted_keys:
        # Performs a True-False check for each amenity in wanted_keys. Booleans are converted to 1-0.
        # Check done: sum(df_list["Kitchen"]) = value_count of Kitchen

        one_hot.append([int(j in i) for i in data["amenities"]])
    
    print(len(one_hot))

    for i, j in zip(wanted_keys, one_hot):
        # Concatenate one-hot with wanted_keys

        data[i] = j

    data.drop("amenities", axis=1, inplace=True)
    return data


def fix_room_type(data):
    """
    Inputs:
    df_list (dataframe): dataframe whose "room_type" will be turned to one hot encoding
    "room_type" (str) : string that specifies "room_type"

    Outputs:

    df_list (dataframe): concatenated dataframe with one hot encoidng of features and dropped previous "room_type"
    """
    data.loc[:, "room_type"] = data["room_type"].apply(lambda x: x.replace(" ", "_").replace("/", "_"))

    # room_types = ["Entire_home_apt", "Hotel_room", "Private_room", "Shared_room"]
    room_types = pd.read_csv('data/room_types.csv').iloc[:, 0]
    data.loc[:, "room_type"] = data.loc[:, "room_type"].astype(pd.CategoricalDtype(categories=room_types))
    dummies = pd.get_dummies(data["room_type"])
    data = pd.concat([data, dummies], axis=1)
    data.drop(columns=["room_type"], inplace=True, axis=1)
    return data


def one_hot_neighbourhood(data):
    neighbourhoods = pd.read_csv('data/neighbourhoods.csv').iloc[:, 0]
    data.loc[:, "neighbourhood_cleansed"] = data.loc[:, "neighbourhood_cleansed"].astype(
        pd.CategoricalDtype(categories=neighbourhoods))
    new_feats = pd.get_dummies(data["neighbourhood_cleansed"])

    data.drop("neighbourhood_cleansed", axis=1, inplace=True)
    return pd.concat([data, new_feats], axis=1)


def generate_categories(df_list):
    # Generate and save neighbourhoods
    neighbourhoods = pd.Series(pd.get_dummies(df_list["neighbourhood_cleansed"]).columns)
    neighbourhoods.to_csv('data/neighbourhoods.csv', index=False)
    # Generate and save room types
    room_types = pd.Series(pd.get_dummies(df_list["room_type"]).columns)
    room_types.to_csv('data/room_types.csv', index=False)
    # Generate and save amenities
    # Create long list of amenities
    flat = [item for sublist in df_list["amenities"] for item in sublist]
    # We will try to group them by amenity and count the most frequent
    from collections import Counter
    # Insert value counts to dataframe
    amenities_df = pd.DataFrame(list(Counter(flat).items())).rename(columns={0: "Amenity", 1: "Count"})

    # Keep most used amenities
    condition = 4000
    amenities_df_pivot = amenities_df[amenities_df["Count"] > condition].sort_values(by="Count",
                                                                                     ascending=False).pivot_table(
        values="Count", columns="Amenity")

    wanted_keys = pd.Series([keys.strip(" ").strip("'") for keys in amenities_df_pivot.columns])

    wanted_keys.to_csv('data/amenities_cat.csv', index=False)


def preprocess(df_list):

    df_list = combine_pre_split_fixes(df_list)

    X_train, X_test, y_train, y_test = split_dataset(df_list)

    X_train, y_train = remove_outliers(X_train, y_train)

    generate_fill_values(X_train)
    X_train = impute_missing_values(X_train)
    X_test = impute_missing_values(X_test)

    save_fit_scaler(X_train)
    X_train = scale_features(X_train)
    X_test = scale_features(X_test)

    generate_categories(df_list)
    # One Hot Encoding of room type
    X_train, X_test = fix_room_type(X_train), fix_room_type(X_test)

    # One-Hot Encoding of most frequent amenities
    X_train, X_test = one_hot_amenities(X_train), one_hot_amenities(X_test)

    # One hot encode neighbourhood
    X_train, X_test = one_hot_neighbourhood(X_train), one_hot_neighbourhood(X_test)

    return X_train, X_test, y_train, y_test


def main():
    df_list = pd.read_csv("data/listings.csv")
    X_train, X_test, y_train, y_test = preprocess(df_list)
    X_train.to_csv("data/X_train.csv", index=False)
    X_test.to_csv("data/X_test.csv", index=False)
    y_train.to_csv("data/y_train.csv", index=False)
    y_test.to_csv("data/y_test.csv", index=False)


if __name__ == "__main__":
    main()
