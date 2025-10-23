from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('src/data/walmart_sales.csv')

def regular_lr():
    # I'm using one-hot encoding to transform the 'Store' categorical variable
    # into separate columns, each for a different store number
    features = pd.get_dummies(
        df[['Store', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']],
        columns=['Store'],
        drop_first=True
    )

    X_train, X_test, y_train, y_test = train_test_split(features, df['Weekly_Sales'], test_size=0.25, random_state=42)

    scaler = StandardScaler()
    scaler.fit(X_train)
    standardized_training = scaler.transform(X_train)
    standardized_test = scaler.transform(X_test)

    lr = LinearRegression()
    lr.fit(standardized_training, y_train)

    y_pred = lr.predict(standardized_test)
    return r2_score(y_test, y_pred)

    # plt.figure(figsize=(10, 6))
    # plt.title('Real values (Blue) vs predictions (Red).')

    # sns.scatterplot(x=X_test['CPI'], y=y_test, color='Blue')
    # sns.scatterplot(x=X_test['CPI'], y=y_pred, color='Red')

    # plt.show()

def polynomial_lr():
    df['Unemployment'] = df['Unemployment'].map(lambda p: p**2)
    df['CPI'] = df['CPI'].map(lambda p: p**2)
    df['Fuel_Price'] = df['Fuel_Price'].map(lambda p: p**3)

    features = pd.get_dummies(
        df[['Store', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']],
        columns=['Store'],
        drop_first=True
    )

    X_train, X_test, y_train, y_test = train_test_split(features, df['Weekly_Sales'], test_size=0.25, random_state=42)

    scaler = StandardScaler()
    scaler.fit(X_train)
    standardized_train = scaler.transform(X_train)
    standardized_test = scaler.transform(X_test)

    lr = LinearRegression()
    lr.fit(standardized_train, y_train)

    y_pred = lr.predict(standardized_test)
    return r2_score(y_test, y_pred)

print(f"regular: {regular_lr()}, poly: {polynomial_lr()}")