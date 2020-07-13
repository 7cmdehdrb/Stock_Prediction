# import tensorflow as tf  # tensorflow
import numpy as np  # Calculate matrix
from keras.models import Sequential
from keras.layers import LSTM, Dense
import pandas as pd  # Load csv file
from matplotlib import pyplot as plt  # Visualize data

# Load Dataset

data = pd.read_csv("src/dataset/005930.KS.csv")
data.head()

# Calculate Average Price

high_prices = data["High"].values
low_prices = data["Low"].values
ave_prices = (high_prices + low_prices) / 2

SEQ_LEN = 50
SEQUENCE_LEN = SEQ_LEN + 1

result = []
for index in range(len(ave_prices) - SEQ_LEN):
    result.append(ave_prices[index:(index + SEQUENCE_LEN)])

# Normalize Data

normalized_data = []
for window in result:
    normalized_window = [(float(p) / float(window[0]) - 1) for p in window]
    normalized_data.append(normalized_window)

result = np.array(normalized_data)

# Split Train and Test Data

row = int(round(result.shape[0] * 0.9))  # shape는 (행, 열) 수를 tuple로 리턴
train = result[:row, :]  # 90%의 행과 전체 열이 트레이닝 셋
np.random.shuffle(train)

x_train = train[:, :-1]  # 마지막 1개를 제외
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))  # 차원 변경
y_train = train[:, -1]  # 마지막 1개

x_test = result[row:, :-1]
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
y_test = result[row:, -1]

print(x_train.shape)
print(y_train.shape)

# Model

model = Sequential()

LSTM_SET = 64
EPOCHS = 20  # 반복 횟수

model.add(LSTM(50, return_sequences=True, input_shape=(50, 1)))
model.add(LSTM(LSTM_SET, return_sequences=False))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='rmsprop')

model.fit(
    x_train,
    y_train,
    validation_data=(x_test, y_test),
    batch_size=10,  # 학습 묶음
    epochs=EPOCHS
)

# Prediction

pred = model.predict(x_test)

fig = plt.figure(facecolor='white')
ax = fig.add_subplot(111)
ax.plot(y_test, label="True")
ax.plot(pred, label="Prediction")
ax.legend()
plt.show()
