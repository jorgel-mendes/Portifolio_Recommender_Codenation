# Lead recommender
Final project from Codenation Data science aceleration, a lead recommender. Based on a existinh market and a portifolio of clients the goal is to recommend possible business leads.

Recommendation is based on how similar the companies are, using a distance algorithm. Then the companies closer to the ones in the portifolio are recommended as possible leads.

The model was constructed with pandas, numpy and scikit-learn. Similarity was calculated through Euclidian distance after a PCA dimensionality reduction.

Results were deployed on google colab, and a small demo of the algorithm was created with IBM Cloud Functions. Both usages are demonstrated in the notebook Recommender Example.

A small sample of the data is on this repository, as well as a portifolio example and a notebook with the EDA process.


## TODO List

This project is also a collection of my studies so improvements will be made as I study more. Probable order of improvements are:

* Creation of a Streamlit App.
* Reorganization of the EDA.
* Remake readme and overall visual.
* Rethink where the model is hosted. I'm not sure cloud function is the best solution for this kind of model.
* Improve the model.
