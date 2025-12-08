import joblib

model = joblib.load('./models/no-pre/kmeans_k2_model.joblib')
print(model.cluster_centers_)
