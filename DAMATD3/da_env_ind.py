import numpy as np
import pandas as pd
import pickle5 as pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from copy import deepcopy

def make_order_book(res_da_info, com_da_info, ind_da_info):
    res_data, com_data, ind_data = deepcopy(res_da_info), deepcopy(com_da_info), deepcopy(ind_da_info)
    for da_info in [res_data, com_data, ind_data]:
        if da_info["da_quantity"] < 0:
            da_info["role"] = "seller"
            da_info["da_quantity"] = -da_info["da_quantity"]
        else:
            da_info["role"] = "buyer"

    df = pd.DataFrame(data=[res_data, com_data, ind_data])
    buy_order_book = df[df.role == "buyer"].sort_values(by='da_price', ascending=False)  # descending price
    sell_order_book = df[df.role == "seller"].sort_values(by='da_price')  # ascending price

    return buy_order_book.reset_index(drop=True), sell_order_book.reset_index(drop=True)


def double_auction_market(res_da_info, com_da_info, ind_da_info, carbon_intensity):

    # ----------------------------- load info ------------------------------- #
    alpha_co2 = 0.0316  # carbon tax $/kg
    tou_price = ind_da_info["tou_price"]
    fit_price = ind_da_info["fit_price"]

    # change price from (-1,1) to real price
    res_da_info["da_price"] = (res_da_info["da_price"] + 1) * 0.5 * (tou_price - fit_price) + fit_price
    com_da_info["da_price"] = (com_da_info["da_price"] + 1) * 0.5 * (tou_price - fit_price) + fit_price
    ind_da_info["da_price"] = (ind_da_info["da_price"] + 1) * 0.5 * (tou_price - fit_price) + fit_price

    # load order book
    buy_order_book, sell_order_book = make_order_book(res_da_info, com_da_info, ind_da_info)
    
    # ----------------------------- market clearing ------------------------------- #
    # pure grid
    # 没有卖家，都是买家，说明自产电不足够，负奖励
    if len(sell_order_book.index) == 0:
        carbon_tax = buy_order_book.da_quantity * carbon_intensity * alpha_co2
        buy_order_book.reward -= buy_order_book.da_quantity * tou_price + carbon_tax
    # 没有买家，说明都产了足够的电，正奖励
    elif len(buy_order_book.index) == 0:
        sell_order_book.reward += sell_order_book.da_quantity * fit_price
    
    
    # mixed da & grid
    # 其他情况
    
    # 买就亏，卖就赚，让order按顺序去满足每个orderbook上的需求，卖完了就换一个microgrid继续卖；买够了就换下一个需要买的microgrid继续买
    else:
        i, j = 0, 0
        while buy_order_book.loc[i, 'da_price'] >= sell_order_book.loc[j, 'da_price']:
            ql = min(buy_order_book.loc[i, 'da_quantity'], sell_order_book.loc[j, 'da_quantity'])
            pl = (buy_order_book.loc[i, 'da_price'] + sell_order_book.loc[j, 'da_price']) / 2
            buy_order_book.loc[i, 'da_quantity'] -= ql
            buy_order_book.loc[i, 'reward'] -= ql * pl
            if buy_order_book.loc[i, 'da_quantity'] == 0:
                i += 1

            sell_order_book.loc[j, 'da_quantity'] -= ql
            sell_order_book.loc[j, 'reward'] += ql * pl
            if sell_order_book.loc[j, 'da_quantity'] == 0:
                j += 1

            if i == len(buy_order_book.index) or j == len(sell_order_book.index):
                break
                
        # balance the unmatched quantity
        # 还没有满足到的部分就用power plant来cover
        carbon_tax = buy_order_book.da_quantity * carbon_intensity * alpha_co2
        buy_order_book.reward -= buy_order_book.da_quantity * tou_price + carbon_tax
        sell_order_book.reward += sell_order_book.da_quantity * fit_price

    # ----------------------------- output reward and order books ------------------------------- #
    df = buy_order_book.append(sell_order_book)
    ind_reward = df[df.name == "ind"].reward.values[0]
#     print('df: ', df)
    return ind_reward


class IndustrialMicrogrid:
    def __init__(self, B_e, B_h2, pes_max, HT_p_max):
        self.n_features_1h = 6  # state: solar,E-H_demand,battery,h2_level,E_price
        self.n_books_1h = 8  # com_ind e_h da price_quantity
        # self.WE_max = 1  # Water Electrolyser input max
        # self.FC_max = 1  # Fuel cell input max
        # self.Etrade_max = 1  # Electricity trading max
        # self.Gtrade_max = 1  # Natural gas trading (m3) max
        self.n_actions = 6  # da_e_price, da_e_quantity, da_h_price, da_h_quantity, WE,FC
        self.battery = 0.0  # battery level
        self.hydrogen = 0.0  # Hydrogen storage level
        self.B_e = B_e  # Battery capacity
        self.B_h2 = B_h2  # Hydrogen storage capacity
        self.pes_max = pes_max  # battery power limit
        self.HT_p_max = HT_p_max  # hydrogen tank flow limit
        self.step_1h = 0
        self.build_MG()

    def build_MG(self):
        with open('data/other_mg_da.pkl', "rb") as fob:
            self.other_mg_da_info = pickle.load(fob)
#         with open('data/res_mg_1hour.pkl', "rb") as fmemg:
#             self.generation_demand_1hour = pickle.load(fmemg)['2019-5-15':'2019-05-16 00:00:00	']
#         with open('data/res_14.pkl', "rb") as fmemg:
#             self.generation_demand_1hour = pickle.load(fmemg)['2014-5-15':'2014-05-16 00:00:00	']
#         with open('data/com_14.pkl', "rb") as fmemg:
#             self.generation_demand_1hour = pickle.load(fmemg)['2014-5-15':'2014-05-16 00:00:00	']
        with open('data/ind_14.pkl', "rb") as fmemg:
            self.generation_demand_1hour = pickle.load(fmemg)['2014-1-15':'2014-12-16 00:00:00	']
        
#         self.generation_demand_1hour = pd.read_pickle('data/res_mg_1hour.pkl')['2019-5-15':'2019-05-16 00:00:00	']

#         self.other_mg_da_info = pd.read_pickle('data/other_mg_da.pkl')
        # self.price = pd.read_pickle('2018price.pkl')['2018-05-16':'2018-05-17 00:00:00	']  # Electricity price ($/Kwh)
        self.price = np.array([0.1199, 0.1199, 0.1199, 0.1199, 0.1199, 0.1199, 0.1199, 0.1199, 0.1199, 0.1199, 0.1199,
                               0.1199, 0.1199, 0.1199, 0.1199, 0.1199, 0.2499, 0.2499, 0.2499, 0.1199, 0.1199, 0.1199,
                               0.1199, 0.1199, 0.1199])
        self.fit_price = 0.04
        self.scaler1 = StandardScaler()
        self.scaler2 = StandardScaler()
        self.scaler3 = StandardScaler()
        self.normalized_gen_dem_1hour = self.scaler1.fit_transform(self.generation_demand_1hour.values)
        self.normalized_mg_da = self.scaler2.fit_transform(self.other_mg_da_info.values)
        self.normalized_price = self.scaler3.fit_transform(self.price.reshape(-1, 1))
        self.gas_price = np.array([0.15820997, 0.17198271, 0.14196519, 0.13808057, 0.1345491, 0.13348966, 0.13313651, 0.12995819, 0.13278337, 0.14267149, 0.15962256, 0.19352469])

        # dollar per cubic meter
        # self.E_trading, self.H_trading = 0, 0  # The energy trading amount

    def reset(self):
        self.battery = 0.0
        self.hydrogen = 0.0
        # self.E_trading = np.random.uniform(-1, 1)
        # self.H_trading = np.random.uniform(-1, 1)
        # reset state (reduce scale for DNN)
        state_1hour = np.hstack((self.normalized_gen_dem_1hour[0, 0],  # solar
                                 self.normalized_gen_dem_1hour[0, 1],  # electricity
                                 self.normalized_gen_dem_1hour[0, 2],  # heat
                                 self.battery / self.B_e, self.hydrogen / self.B_h2,
                                 self.normalized_price[0, 0]))

        public_book = np.hstack((self.other_mg_da_info.iloc[0, 0],  # com e price
                                 self.normalized_mg_da[0, 1],  # com e quantity
                                 self.other_mg_da_info.iloc[0, 2],  # com h price
                                 self.normalized_mg_da[0, 3],  # com h quantity
                                 self.other_mg_da_info.iloc[0, 4],  # ind e price
                                 self.normalized_mg_da[0, 5],  # ind e quantity
                                 self.other_mg_da_info.iloc[0, 6],  # ind h price
                                 self.normalized_mg_da[0, 7]  # ind h quantity
                                 ))

        # reset step
        self.step_1h = 0
        return state_1hour, public_book

    def state_attacked(self, attack_level, attack_clip):
        attack_signal = np.random.normal(0, attack_level)
        attack_signal = np.clip(attack_signal, -attack_clip, attack_clip)
        attacked_demand = self.normalized_gen_dem_1hour[self.step_1h, 1] + attack_signal  # electricity demand

        attacked_state = np.hstack((self.normalized_gen_dem_1hour[self.step_1h, 0], attacked_demand,
                                    self.normalized_gen_dem_1hour[self.step_1h, 2],
                                    self.battery / self.B_e, self.hydrogen / self.B_h2,
                                    self.normalized_price[self.step_1h, 0]))

        return attacked_state

    def sample_action(self):
        # random select trading actions
        # y_elec = np.random.uniform(0, self.Etrade_max)  # *2-1==>(-1, 1) sigmoid to tanh
        # y_gas = np.random.uniform(0, self.Gtrade_max)
        # a_WE = np.random.uniform(0, self.WE_max)
        # a_FC = np.random.uniform(0, self.FC_max)
        #
        # return np.array([y_elec, y_gas, a_WE, a_FC])
        actions = np.random.uniform(-1, 1, self.n_actions)
        return actions
    
    
    def sample_trading(self):
        with open('data/other_mg_da.pkl', "rb") as fob:
            trading_a = pickle.load(fob)
        return trading_a

    def convert_energy(self, action):
        # -----------------------parameters----------------------------
        tou_price = self.price[self.step_1h]  # electricity buy price from the grid
        fit_price = self.fit_price  # electricity sell price
        p_ng = self.gas_price[4]  # natural gas price (May)
        p_h2 = 5  # hydrogen gas price ($/kg)

        # s_bg = 1.2  # price ratio of buying from grid to p2p
        # s_sg = 0.8  # price ratio of selling from grid to p2p

        k_we = 0.8  # water electrolyser efficiency
        k_fc_e = 0.3  # fuel cell to electricity efficiency
        k_fc_h = 0.55  # fuel cell to heat efficiency
        k_gb = 0.9  # gas boiler efficiency
        k_ng2q = 8.816  # natural gas(m3) to Q(KWh) ratio
        k_h2q = 33.33  # hydrogen(kg) to Q(KWh) ratio

        c_p = 3 * tou_price  # electricity penalty coefficient
        c_h = 3 * p_ng  # heat penalty coefficient

        beta_gas = 0.245  # carbon intensity kg/kwh
        beta_elec = 0.683  # carbon intensity kg/kwh

        reward = 0
        E_dif = 0  # electricity balance difference (Kwh)
        h2_dif = 0  # h2 balance difference (kg)
        H_dif = 0  # heat balance difference (m3)

        # ----------------------------- DA market ------------------------------- #

        # da market quantity bid/offer
        E_da_quantity = action[1] * 100  # kwh
        H_da_quantity = action[3] * 100  # kwh
        
        # 要改就把需要调度的那个MEMG的写成action
        ind_da_elec = {
            "name": "ind",
            "da_price": action[0], "tou_price": tou_price, "fit_price": fit_price,
            "da_quantity": E_da_quantity, "reward": 0}
        res_da_elec = {
            "name": "res",
            "da_price": self.other_mg_da_info.iloc[self.step_1h, 0],
            "da_quantity": self.other_mg_da_info.iloc[self.step_1h, 1], "reward": 0}
        com_da_elec = {
            "name": "com",
            "da_price": self.other_mg_da_info.iloc[self.step_1h, 4],
            "da_quantity": self.other_mg_da_info.iloc[self.step_1h, 5], "reward": 0}

        e_reward = double_auction_market(res_da_elec, com_da_elec, ind_da_elec, beta_elec)
        reward += e_reward  # res da e reward

        tou_price_h = self.gas_price[4] / (k_ng2q * k_gb)  # equivalent heat price $/kwh (heat provided by gas boiler)
        fit_price_h = 0  # can not sell heat back to the external network
        ind_da_heat = {
            "name": "ind",
            "da_price": action[2], "tou_price": tou_price_h, "fit_price": fit_price_h,
            "da_quantity": H_da_quantity, "reward": 0}
        res_da_heat = {
            "name": "res",
            "da_price": self.other_mg_da_info.iloc[self.step_1h, 2],
            "da_quantity": self.other_mg_da_info.iloc[self.step_1h, 3], "reward": 0}
        com_da_heat = {
            "name": "com",
            "da_price": self.other_mg_da_info.iloc[self.step_1h, 6],
            "da_quantity": self.other_mg_da_info.iloc[self.step_1h, 7], "reward": 0}

        h_reward = double_auction_market(res_da_heat, com_da_heat, ind_da_heat, beta_gas)
        reward += h_reward  # res da h reward

        # ---------------------energy conversion----------------------

        a_WE = (action[4] + 1) * 0.5 * 200  # kwh
        a_FC = (action[5] + 1) * 0.5 * 8  # kg

        E_FC = a_FC * k_h2q * k_fc_e  # electricity (kwh) output from FC
        H_FC = a_FC * k_h2q * k_fc_h  # heat (kwh) output from FC
        h2_WE = a_WE * k_we / k_h2q  # hydrogen output (kg) from WE

        # ---------------------energy balance-------------------------
        solar = self.generation_demand_1hour.iloc[self.step_1h, 0]  # solar output
        E_demand = self.generation_demand_1hour.iloc[self.step_1h, 1]  # electricity load
        H_demand = self.generation_demand_1hour.iloc[self.step_1h, 2]   # heat load
        # reward += pL1 * pp * 1.8  # retail profit

        # battery charging (if>0) amount by electricity balance
        E_battery = solar + E_da_quantity + E_FC - a_WE - E_demand
        if E_battery < -self.pes_max:  # battery level and penalty
            self.battery += -self.pes_max / 0.9
            E_dif += -self.pes_max - E_battery
        if -self.pes_max <= E_battery < 0:
            self.battery += E_battery / 0.9
        if 0 < E_battery <= self.pes_max:
            self.battery += E_battery * 0.9
        if E_battery > self.pes_max:
            self.battery += self.pes_max * 0.9  # no penalty as described in the paper
        if self.battery > self.B_e:
            self.battery = self.B_e  # no penalty as described in the paper
        if self.battery < 0:
            E_dif += -self.battery
            self.battery = 0

        h2_HT = h2_WE - a_FC  # hydrogen tank inflow amount (if>0) kg by hydrogen balance
        if h2_HT < -self.HT_p_max:  # hydrogen tank level and cost
            self.hydrogen += -self.HT_p_max / 0.95
            h2_dif += -self.HT_p_max - h2_HT
        if -self.HT_p_max <= h2_HT < 0:
            self.hydrogen += h2_HT / 0.95
        if 0 < h2_HT <= self.HT_p_max:
            self.hydrogen += h2_HT * 0.95
        if h2_HT > self.HT_p_max:
            self.hydrogen += self.HT_p_max * 0.95
            # don't sell surplus hydrogen as we don't want to use hydrogen in this way
        if self.hydrogen > self.B_h2:
            self.hydrogen = self.B_h2  # as above
        if self.hydrogen < 0:
            h2_dif += -self.hydrogen
            self.hydrogen = 0

        if H_da_quantity + H_FC < H_demand:  # heat balance
            H_dif += (H_demand - H_da_quantity - H_FC) / k_ng2q

        # -------------------calculating reward-----------------------

        reward -= E_dif * c_p + ((H_dif/10) ** 2 + H_dif) * c_h  # penalty
        # reward -= E_dif * c_p + H_dif * c_h  # penalty
        reward -= h2_dif * p_h2  # buy needed hydrogen

        # ------------------------next state--------------------------
        self.step_1h += 1  # next step

        s1h_ = np.hstack((self.normalized_gen_dem_1hour[self.step_1h, 0],
                          self.normalized_gen_dem_1hour[self.step_1h, 1],
                          self.normalized_gen_dem_1hour[self.step_1h, 2],
                          self.battery / self.B_e, self.hydrogen / self.B_h2,
                          self.normalized_price[self.step_1h, 0]))

        # next public book
        public_ = np.hstack((self.other_mg_da_info.iloc[self.step_1h, 0],  # com e price
                             self.normalized_mg_da[self.step_1h, 1],  # com e quantity
                             self.other_mg_da_info.iloc[self.step_1h, 2],  # com h price
                             self.normalized_mg_da[self.step_1h, 3],  # com h quantity
                             self.other_mg_da_info.iloc[self.step_1h, 4],  # ind e price
                             self.normalized_mg_da[self.step_1h, 5],  # ind e quantity
                             self.other_mg_da_info.iloc[self.step_1h, 6],  # ind h price
                             self.normalized_mg_da[self.step_1h, 7]  # ind h quantity
                             ))


        # return reward / 10, s15m_, s1h_, s1h_15m_3np2r, E_battery  # for load prediction
        return reward, s1h_, public_, e_reward, h_reward
