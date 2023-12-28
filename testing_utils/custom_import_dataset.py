from river.stream import iter_csv

def read_covertype():
    return iter_csv("datasets/covertype/covertype.csv", target="54", converters={f"{i}": int for i in range(54)})
    
    
def get_iter_stream(dataset_name):
    if dataset_name == "covertype":
        return read_covertype()