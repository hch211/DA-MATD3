import numpy as np
from collections import deque
import tensorflow as tf
from res_da_env import ResidentialMicrogrid
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def create_data_for_regression():
    np.random.seed(42)
    env = ResidentialMicrogrid(B_e=300, B_h2=30, pes_max=25, HT_p_max=3)
    td3_1h = tf.keras.models.load_model("checkpoint/1h_model.h5")
#     td3_15min = tf.keras.models.load_model("checkpoint/1h_model.h5")
#     td3_15min = tf.keras.models.load_model("15min_model.h5")
    data_list = deque()
    battery_list = []
    rewards_list = []

    for test_episode in range(100):
        s_1h, _ = env.reset()
        trading_a = env.sample_trading()
        rewards = 0

        for i in range(24):
            solar = env.generation_demand_1hour.iloc[i, 0]
            conversion_a = td3_1h(np.array([s_1h])).numpy()[0]
            E_demand = env.generation_demand_1hour.iloc[i, 1]

            # Step the env
#             print('trading: ', trading_a)
#             print('conversion: ', conversion_a)
            reward, s2_1h, E_battery = env.convert_energy(conversion_a)
#             print('s2: ', s2_1h)
            rewards += reward
            battery_1, battery_2 = s2_1h[3], s_1h[3]
            s_1h = s2_1h

            # store the data y_elec, a_WE, a_FC
            if i > 3:
                data_list.append([battery_1, battery_2, solar, trading_a[0], conversion_a[0], conversion_a[1], E_demand])

            if i % 4 == 3:
                s_1h = s2_1h
                trading_a = td3_1h(np.array([s_1h])).numpy()[0]
        rewards_list.append(rewards)

    data_for_regression = np.array(data_list)
    input_data = data_for_regression[:, :-1]
    target = data_for_regression[:, -1]
    return input_data, target, battery_list, rewards_list


if __name__ == "__main__":
    X, y, E_battery, rewards = create_data_for_regression()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    
    # 线性回归拟合的预测值
    y_predict = lin_reg.predict(X_test)
    
    # MAE
    mae = np.mean(np.abs(y_predict-y_test)/y_test)
    print(f'Test Accuracy is {1-mae}')
