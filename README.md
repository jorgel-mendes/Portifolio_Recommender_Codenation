# Portifolio_Recommender_Codenation
Final project from Codenation Data science aceleration, a portifolio recommender


Esta é a minha solução para o desafio final da codenation, com um algoritmo de recomendação para possíveis leads.

A recomendação é feita baseada na similaridade das empresas. São alimentados algumas empresas do portifólio do cliente para o modelo e então empresas mais próximas são indicadas como possíveis leads.

O modelo utiliza as bibliotecas pandas, numpy scikit-learn e para pré processamento e criação do modelo. A similaridade é calculada através de distância Euclidiana, após uma dimensão de dimensionalidade por PCA.

Os resultados foram feitos no Google Colab por limitações de hardware.

No notebook example são utilizados os dados do portifólio 2 para demonstração do problema, utilizando o modelo de Modelo_desafio_final.py.

A análise exploratória com um maior estudo sobre as variáveis e o baseline do modelo se encontram em EDA.ipynb
