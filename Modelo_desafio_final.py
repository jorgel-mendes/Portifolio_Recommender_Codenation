#Modelo para desafio final codenation

import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MaxAbsScaler
from sklearn.decomposition import PCA

def get_recommendations(portifolio, n_recommendations, market_initial):

    # Retirando dados de idenficação
    market_reduced = market_initial.drop(['Unnamed: 0', 'id'], axis=1)

    #Retirando dados com muitos NAs
    isna_columns = market_reduced.isna().sum().sort_values(ascending=False)
    isna_columns = isna_columns[isna_columns > 462298*.9].index
    market_reduced.drop(isna_columns, axis=1, inplace=True)

    #Retirando dados com pouca variabilidade
    market_counts = market_initial.apply(lambda x: pd.value_counts(x).max())
    market_reduced.drop(market_counts[market_counts > 462298*.90].index, axis = 1, inplace=True)

    #retirando colunas por análise do EDA
    to_drop = ['de_natureza_juridica', 'de_ramo', 'nm_divisao', 
               'dt_situacao', 'nm_micro_regiao', 
               'sg_uf_matriz','de_faixa_faturamento_estimado_grupo']


    market_reduced.drop(to_drop, axis=1, inplace=True)


    #Seleciona as vairáveis categóricas
    categorical_cols = [cname for cname in market_reduced.columns if 
                        market_reduced[cname].nunique() < 15 and 
                        market_reduced[cname].dtype == "object"]

    # Seleciona as variáveis numéricas
    numerical_cols = [cname for cname in market_reduced.columns if 
                    market_reduced[cname].dtype in ['int64', 'bool', 'float64']]

    # Preprocessamento de variáveis numéricas
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # Preprocessamento de dados categóricos
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MaxAbsScaler())
    ])

    # Pipeline com processamentos
    preprocessor_pipeline = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])


    pca_transformer = PCA(n_components=.6)
    full_pipeline = Pipeline(steps=[('preprocessor', preprocessor_pipeline),
                          ('pca', pca_transformer)
                         ])

    #Substituindo NAs das variáveis categóricas com a moda
    market_reduced[categorical_cols] = market_reduced[categorical_cols].fillna(
        value=market_reduced[categorical_cols].mode().iloc[0])

    #Seleecionando as observações do portifólio utilizadas para o treino
    selected_companies = portifolio.iloc[:, 1]
    test_indexes = market_initial[market_initial['id'].isin(selected_companies)].index

    pca_model = full_pipeline.fit(market_reduced.iloc[test_indexes])
    pca_train = pca_model.transform(market_reduced)
    pca_test = pca_model.transform(market_reduced.iloc[test_indexes])
    
    #Calulando distância
    distance_result = [pd.Series(np.linalg.norm((pca_train - pca_test[test_index]), axis=1)) for test_index in range(len(pca_test))]
    distance_result_df = pd.concat(distance_result, axis=1)
    distance_result_df.drop(test_indexes, inplace=True)
    
    result_rank = distance_result_df.rank().min(axis=1).sort_values()
    result_indexes = result_rank[0:n_recommendations].index
    
    return market_initial.iloc[result_indexes, :]