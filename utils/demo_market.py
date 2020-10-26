#Code to create the market demo used in the streamlit app.

import pandas as pd
from requests import get
from zipfile import ZipFile
from io import BytesIO

def get_demo_market():
    remotezip = get("https://codenation-challenges.s3-us-west-1.amazonaws.com/ml-leads/estaticos_market.csv.zip")
    root = ZipFile(BytesIO(remotezip.content))
    market_initial =  pd.read_csv(root.open('estaticos_market.csv')) 

    portifolio2 = pd.read_csv("https://codenation-challenges.s3-us-west-1.amazonaws.com/ml-leads/estaticos_portfolio2.csv")

    test_indexes = market_initial[market_initial['id'].isin(portifolio2['id'])].index

    market_demo = pd.concat([market_initial.iloc[test_indexes],
               market_initial.drop(test_indexes).sample(n=2*len(portifolio2), 
                                                        replace=False, 
                                                        random_state = 1)]
            )
    # market_demo.to_csv("market_demo.csv")
    
    return market_demo

