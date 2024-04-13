import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM


def neurone():
    # Загружаем исторические данные о цене акции
    data = pd.read_csv('stock_price.csv')

    # Масштабируем данные в диапазоне от 0 до 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Price'].values.reshape(-1, 1))

    # Создаем набор данных для обучения и тестирования
    train_size = int(len(scaled_data) * 0.8)
    test_size = len(scaled_data) - train_size
    train_data, test_data = scaled_data[0:train_size, :], scaled_data[train_size:len(scaled_data), :]

    # Преобразуем данные в формат, подходящий для обучения нейросети
    x_train = []
    y_train = []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60:i, 0])
        y_train.append(train_data[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Создаем нейросеть с использованием LSTM-слоев
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))

    # Обучаем нейросеть
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

    # Прогнозируем цену акции на основе тестовых данных
    x_test = []
    y_test = data['Price'][train_size:].values
    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60:i, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predicted_price = model.predict(x_test)
    predicted_price = scaler.inverse_transform(predicted_price)

    # Выводим предсказанную цену акции
    print(f'Predict: {predicted_price}')


def generation():
    ### Start generation ###

    # Создаем функцию для генерации искусственных данных о цене акции
    def generate_stock_data(start_date, end_date, start_price, volatility):
        dates = pd.date_range(start_date, end_date)
        prices = []
        current_price = start_price
        for i in range(len(dates)):
            price_change = np.random.normal(0, volatility)
            current_price += price_change
            prices.append(current_price)
        data = pd.DataFrame({'Date': dates, 'Price': prices})
        return data

    # Генерируем искусственные данные о цене акции
    start_date = '2020-01-01'
    end_date = '2022-12-31'
    start_price = 100
    volatility = 0.01
    stock_data = generate_stock_data(start_date, end_date, start_price, volatility)

    # Сохраняем данные в CSV-файл
    stock_data.to_csv('stock_price.csv', index=False)

    ### End generation ###


if __name__ == '__main__':
    neurone()