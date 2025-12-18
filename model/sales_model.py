from sklearn.linear_model import LinearRegression

def train_model(df):
    monthly_revenue = df.groupby("month")["revenue"].sum().reset_index()
    X = monthly_revenue[["month"]]
    y = monthly_revenue["revenue"]

    model = LinearRegression()
    model.fit(X, y)
    return model

def predict_revenue(model, month):
    return model.predict([[month]])[0]
