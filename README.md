# Supply Chain MLOps - Client reviews unsupervised clustering 

This project stems from a collective work produced in the context of Liora Machine Learning engineering formation. The current version of it has been reframed entirely for learning purposes.
This work has been both an organizational and technical challenge. Both aspects are documented in the present README file.

## Technicalities

The main purpose was to implement the complete MLops chain from project design up to production and monitoring of the application and models. The starting point was 40 000 scrapped Trustpilot reviews.
We aimed at implementing a clustering tool enabling automatic reviews clustering coupled with sentiment analysis. This reframed version of the project is focused on unsupervised semantic clustering only.
The main point is to display the stability and structure of the whole MLops chain. 

### Rebuild step 1: Exploratory phase

Four pipelines were explored during this exploratory phase. Using SentenceTransformers paraphrase-multilingual-mpnet-base-v2 to generate embeddings (with L2 normalization):
* KMeans
* Spherical KMeans
* HDBSCAN
* UMAP dimensional reduction followed by HDBSCAN

All pipelines were tracked using MLflow. Geometric, semantic, statistical and algorithmic coherence and stability have been investigated.
 
