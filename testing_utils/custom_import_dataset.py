from river.stream import iter_pandas
import gzip
from scipy.io.arff import loadarff 
import pandas as pd

def read_covertype():
    with gzip.open("./datasets/covertype/covtype.data.gz") as f:
        df = pd.read_csv(f, sep=',')
    label_col = df.columns[-1]
    feature_cols = list(df.columns)
    feature_cols.pop()
    X = df[feature_cols]
    Y = df[label_col]
    return iter_pandas(X=X, y=Y)
    
def read_others(dataset_name):
    raw_data = loadarff("./datasets/"+dataset_name+".arff")
    df = pd.DataFrame(raw_data[0])
    label_col = df.columns[-1]
    feature_cols = list(df.columns)
    feature_cols.pop()
    X = df[feature_cols]
    Y = df[label_col]
    return iter_pandas(X=X, y=Y)

    
def get_iter_stream(dataset_name):
    if dataset_name == "covertype":
        return read_covertype()
    else:
        return read_others(dataset_name)