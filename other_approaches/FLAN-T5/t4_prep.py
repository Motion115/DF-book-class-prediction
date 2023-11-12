from utils import load_data
from models.GCN import GCN
import json
if __name__ == "__main__":
    data_preprocessor, model = load_data("./data/bert-cls-embeddings.pth", GCN)
    data = data_preprocessor.raw_data_source

    categories = data[["category"]]
    cats = set()
    for i in range(len(categories)):
        val = categories["category"][i]
        cats.add(val)
    print(cats)




    # preserve only the text column
    texts = data[["text"]]
    data = texts["text"]




    # # traverse data, count for ; as delimiter
    # for key, value in data.items():
    #     # match for keywords Description and Title
    #     if "Description" not in value:
    #         print(key)
    #     if "Title" not in value:
    #         print(key)
            
    
    exit()
    # convert to json
    data = data.to_dict("dict")
    # save data as a json file
    with open("./data/texts.json", "w") as f:
        json.dump(data, f, indent=2)