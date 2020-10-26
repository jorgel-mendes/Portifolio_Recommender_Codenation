#TODO: Profile to reduce memory usage

import pandas as pd
import numpy as np
import json

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MaxAbsScaler
from sklearn.decomposition import PCA

def main(portifolio):
    """Get n leads recommendations from a certain marketing based on a provided portifolio.
    
    Arguments:
    portifolio -- Data frame with user's portifolio. Companies in the portifolio should be in the market Data frame.
    n_recommendations -- Number of recommendations returned.
    market_initial -- Data frame with companies in the market.
    """
    from requests import get
    from zipfile import ZipFile
    from io import BytesIO

    remotezip = get("https://codenation-challenges.s3-us-west-1.amazonaws.com/ml-leads/estaticos_market.csv.zip")
    root = ZipFile(BytesIO(remotezip.content))
    market_initial =  pd.read_csv(root.open('estaticos_market.csv')) 
    
    n_recommendations = portifolio["n_recommendations"]
    portifolio = pd.DataFrame({'id': portifolio['id']})
    
    # Dropping columns with ids
    market_reduced = market_initial.drop(['Unnamed: 0', 'id'], axis=1)
    
    del remotezip
    del root
    # Dropping columns with more than 90% NAs
    isna_columns = market_reduced.isna().sum().sort_values(ascending=False)
    return portifolio.to_dict()
    isna_columns = isna_columns[isna_columns > 462298*.9].index
    market_reduced.drop(isna_columns, axis=1, inplace=True)
    
    # Dropping columns with  more than 90% with the same category
    market_counts = market_initial.apply(lambda x: pd.value_counts(x).max())
    market_reduced.drop(market_counts[market_counts > 462298*.90].index, axis = 1, inplace=True)
    
    # Columns dropped after EDA analysis
    to_drop = ['de_natureza_juridica', 'de_ramo', 'nm_divisao', 
               'dt_situacao', 'nm_micro_regiao', 
               'sg_uf_matriz','de_faixa_faturamento_estimado_grupo']
    market_reduced.drop(to_drop, axis=1, inplace=True)

    #Variables for preprocessing
    categorical_cols = [cname for cname in market_reduced.columns if 
                        market_reduced[cname].nunique() < 15 and 
                        market_reduced[cname].dtype == "object"]
    numerical_cols = [cname for cname in market_reduced.columns if 
                    market_reduced[cname].dtype in ['int64', 'bool', 'float64']]

    # Preprocessing pipeline
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MaxAbsScaler())
    ])
    preprocessor_pipeline = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    # Adding pca to the pipeline
    pca_transformer = PCA(n_components=.6)
    full_pipeline = Pipeline(steps=[('preprocessor', preprocessor_pipeline),
                          ('pca', pca_transformer)
                         ])
    
    # NA replacement for categorical features is made with pandas for efficiency
    market_reduced[categorical_cols] = market_reduced[categorical_cols].fillna(
        value=market_reduced[categorical_cols].mode().iloc[0])

    # Defining the values with the portifolio companies
    selected_companies = portifolio['id']
    test_indexes = market_initial[market_initial['id'].isin(selected_companies)].index
    
    # Preprocessing and pca transformation
    pca_model = full_pipeline.fit(market_reduced.iloc[test_indexes])
    market_transformed = pca_model.transform(market_reduced)
    portifolio_transformed = pca_model.transform(market_reduced.iloc[test_indexes])
    
    # Calculating the distance data frame
    distance_result = [pd.Series(np.linalg.norm((market_transformed - portifolio_transformed[test_index]), axis=1)) 
    for test_index in range(len(portifolio_transformed))]

    distance_result_df = pd.concat(distance_result, axis=1)
    distance_result_df.drop(test_indexes, inplace=True)
    
    # Selecting 
    result_rank = distance_result_df.rank().min(axis=1).sort_values()
    result_indexes = result_rank[0:n_recommendations].index
    
    return market_initial.iloc[result_indexes, :].to_dict()
