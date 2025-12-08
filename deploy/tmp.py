import joblib

model = joblib.load('./models/no-pre/gmm_model.pkl')
print(model.keys())
scaler = model['scaler']
joblib.dump(scaler, './models/no-pre/gmm_scaler.joblib')
