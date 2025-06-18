# library imports
import re
import pandas as pd
import yfinance as YF
import mysql.connector as SQL
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder as LE
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as DTC, plot_tree as PT
from sklearn.metrics import accuracy_score, classification_report


# make data prep class
class SimpleDataProcessing:

    # initialize the class
    def __init__(self, file_path):
        self.file_path = file_path
        self.label_encoders = {}

    # read in data
    def load_data(self):
        df = pd.read_csv(self.file_path)
        return df
    
    # transform categorical columns
    def transform_cat_cols(self, df):
        df = df.copy()
        for col in df.select_dtypes(include=["object"]).columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LE()
                df[col] = self.label_encoders[col].fit_transform(df[col])
            else:
                df[col] = self.label_encoders[col].transform(df[col])
        return df
    
    # split data
    def make_XY(self, df, target_var):
        df = self.transform_cat_cols(df)

        # drop ID columns, isolate target var
        y = df[[target_var]]
        X = df.drop(columns=[target_var])
        X = X.drop(columns=[col for col in X.columns if re.match(r".*ID", col)], errors='ignore')

        # make x and y sets
        x_train, x_temp, y_train, y_temp = train_test_split(X, y, test_size=.2, random_state=42)
        x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=.5, random_state=42)
        return x_train, x_val, x_test, y_train, y_val, y_test
    
    # make and train model
    def make_and_train_tree(self, x_train, y_train, max_depth):
        model = DTC(random_state=42, max_depth=max_depth)
        model.fit(x_train, y_train)
        return model
    
    # plot the tree
    def plot_tree(self, model, x_train, class_names=['Approved', 'Rejected']):
        plt.figure(figsize=(15, 10))
        PT(model, filled=True, feature_names=x_train.columns, class_names=class_names)
        plt.show()

    # evaluate model
    def evaluate_model(self, model, x_test, y_test):
        y_pred = model.predict(x_test)
        print(f"Accuracy score: {accuracy_score(y_test, y_pred)}")
        print(classification_report(y_test, y_pred))

    # preprocess
    def process_data(self, target_var):
        df = self.load_data()
        return self.make_XY(df, target_var)


# func to obtain stock data info from yahoo finance
def get_tickers(tickers=[], period='1d'):
    # get tickers from yahoo finance
    table = pd.DataFrame()
    for ticker in tickers:
        data = YF.Ticker(ticker)
        df = data.history(period=period)
        df['ticker'] = ticker
        table = pd.concat([table, df])
    return table


# import data from mysql db into a df
def SQL2DF(db, table, drop_id=False):
    conn = SQL.connect(host='localhost',
                        user='root',
                        password='root',
                        db=db)

    cursor = conn.cursor()
    cursor.execute(f"""SELECT * FROM {table}""")
    data = cursor.fetchall()
    cols = [desc[0] for desc in cursor.description]
    df = pd.DataFrame(data, columns=cols)
    if drop_id:
        df.drop(columns=['id'], inplace=True)
    return df
