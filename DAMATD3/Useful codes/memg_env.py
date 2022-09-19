import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
# states_info = pd.read_excel('some_datasheet.xlsx', index_col=0)  # external states related to time


def initialize_external_states(states_info):  # load data from external resources
    scaler = MinMaxScaler()
    normalized_states_info = scaler.fit_transform(states_info.values)
    return states_info, normalized_states_info


def make_order_book(res_da_info, com_da_info, ind_da_info):
    from copy import deepcopy
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


def double_auction_market(res_da_info, com_da_info, ind_da_info):

    # ----------------------------- load info ------------------------------- #
    # make sure the time steps of MGs are equal
    assert res_da_info["tou_price"] == com_da_info["tou_price"] == ind_da_info["tou_price"]
    tou_price = res_da_info["tou_price"]
    fit_price = res_da_info["fit_price"]

    # for centralised critic (others' da price and da quantity) NEED NORMALIZE!!
    res_critic_da_info = [com_da_info["da_price"], com_da_info["da_quantity"]/200,
                          ind_da_info["da_price"], ind_da_info["da_quantity"]/200]
    com_critic_da_info = [res_da_info["da_price"], res_da_info["da_quantity"]/200,
                          ind_da_info["da_price"], ind_da_info["da_quantity"]/200]
    ind_critic_da_info = [com_da_info["da_price"], com_da_info["da_quantity"]/200,
                          res_da_info["da_price"], res_da_info["da_quantity"]/200]

    # change price from (-1,1) to real price
    res_da_info["da_price"] = (res_da_info["da_price"] + 1) * 0.5 * (tou_price - fit_price) + fit_price
    com_da_info["da_price"] = (com_da_info["da_price"] + 1) * 0.5 * (tou_price - fit_price) + fit_price
    ind_da_info["da_price"] = (ind_da_info["da_price"] + 1) * 0.5 * (tou_price - fit_price) + fit_price

    # load order book
    buy_order_book, sell_order_book = make_order_book(res_da_info, com_da_info, ind_da_info)

    # ----------------------------- market clearing ------------------------------- #
    # pure grid
    if len(sell_order_book.index) == 0:
        buy_order_book.reward -= buy_order_book.da_quantity * tou_price

    elif len(buy_order_book.index) == 0:
        sell_order_book.reward += sell_order_book.da_quantity * fit_price
    # mixed da & grid
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
        buy_order_book.reward -= buy_order_book.da_quantity * tou_price
        sell_order_book.reward += sell_order_book.da_quantity * fit_price

    # ----------------------------- output reward and order books ------------------------------- #
    df = buy_order_book.append(sell_order_book)
    res_final = {"reward": df[df.name == "res"].reward.values[0]/100,
                 "public_info": res_critic_da_info}
    com_final = {"reward": df[df.name == "com"].reward.values[0]/100,
                 "public_info": com_critic_da_info}
    ind_final = {"reward": df[df.name == "ind"].reward.values[0]/100,
                 "public_info": ind_critic_da_info}

    return res_final, com_final, ind_final


class ResidentialMicrogrid:
    def __init__(self, e_ees, p_ees, e_tes, p_tes, gas_fc_max, gas_gb_max, states_info, mg_node_max=1e6):

        self.n_states = 6  # no. of states
        self.n_actions = 5  # no. of actions (fc, gb, ees, hss, da price)
        # all action max is 1

        # system variable
        self.E_ees_max = e_ees  # EES capacity
        self.P_ees_max = p_ees  # EES power capacity
        self.E_tes_max = e_tes  # TES capacity
        self.P_tes_max = p_tes  # TES power capacity
        self.gas_fc_max = gas_fc_max  # FC input capacity (kwh)
        self.gas_gb_max = gas_gb_max  # GB input capacity (kwh)
        self.mg_node_max = mg_node_max  # the max quantity can be submitted into the da market

        # internal states
        self.S_ees = 0.1  # SOC of EES
        self.S_tes = 0.1  # thermal storage level
        # external states dataframe
        self.states_info, self.normalized_states_info = initialize_external_states(states_info)
        self.fit_price = 0.04
        self.gas_price = 0.0338  # $/kwh
        # keep track of step for loading external states
        self.step = 0  # time step

    def reset(self):
        self.S_ees = 0.1
        self.S_tes = 0.1
        normalised_pv = self.normalized_states_info[0, 0]
        normalised_e_load = self.normalized_states_info[0, 1]
        normalised_h_load = self.normalized_states_info[0, 2]
        normalised_tou_price = self.normalized_states_info[0, 3]
        initial_states = np.hstack((self.S_ees, self.S_tes, normalised_pv,
                                    normalised_e_load, normalised_h_load, normalised_tou_price))  # reset state
        self.step = 0  # reset step
        return initial_states

    def sample(self):
        # random select actions [action_fc, action_gb, action_ees, action_hss, action_da_price]
        actions = np.random.uniform(-1, 1, self.n_actions)
        return actions

    def env_step(self, action):
        # ---------------------- in scope parameters -------------------- #

        k_fc_e = 0.3  # fuel cell to electricity efficiency
        k_fc_h = 0.55  # fuel cell to heat efficiency
        k_gb = 0.8  # gas boiler efficiency

        # wc_ees = 0.0091  # battery wear cost ($/kwh)
        k_ees = 0.95  # EeS charging/discharging efficiency
        k_tes = 0.9  # TES charging/discharging efficiency

        beta_gas = 0.245  # carbon intensity kg/kwh
        alpha_co2 = 0.0316  # carbon tax $/kg 0.0316

        reward = 0
        e_dif = 0

        # ----------------------- load external states ------------------ #
        pv_true = self.states_info.iloc[self.step, 0]
        e_load_true = self.states_info.iloc[self.step, 1]
        h_load_true = self.states_info.iloc[self.step, 2]
        tou_price = self.states_info.iloc[self.step, 3]
        fit_price = self.fit_price
        gas_price = self.gas_price
        c_p_e = 3 * tou_price  # electricity penalty coefficient
        c_p_h = 3 * gas_price  # heat penalty coefficient

        # ------------------------- load actions ---------------------- #
        g_fc = (action[0]+1) * 0.5 * self.gas_fc_max  # FC input (kwh)
        e_fc = g_fc * k_fc_e
        h_fc = g_fc * k_fc_h

        g_gb = (action[1]+1) * 0.5 * self.gas_gb_max  # GB input (m3)
        h_gb = g_gb * k_gb

        e_ees = action[2] * self.P_ees_max  # EES input electricity (kwh) >0 charge otherwise discharge
        h_tes = action[3] * self.P_tes_max  # TES input heat (kwh) >0 charge otherwise discharge

        # --------------------- calculate next internal state ---------------- #
        # EES
        if e_ees < 0:
            self.S_ees += (e_ees / self.E_ees_max) / k_ees
            if self.S_ees < 0.1:
                e_ees += (0.1 - self.S_ees) * self.E_ees_max * k_ees
                self.S_ees = 0.1
        else:
            self.S_ees += (e_ees / self.E_ees_max) * k_ees
            if self.S_ees > 1:
                e_ees -= (self.S_ees - 1) * self.E_ees_max / k_ees
                self.S_ees = 1
        # TES
        if h_tes < 0:
            self.S_tes += (h_tes / self.E_tes_max) / k_tes
            if self.S_tes < 0.1:
                h_tes += (0.1 - self.S_tes) * self.E_tes_max * k_tes
                self.S_tes = 0.1
        else:
            self.S_tes += (h_tes / self.E_tes_max) * k_tes
            if self.S_tes > 1:
                h_tes -= (self.S_tes - 1) * self.E_tes_max / k_tes
                self.S_tes = 1

        # ----------------- calculate da market info and partial reward ---------------- #
        co2_emission = (g_fc + g_gb) * beta_gas
        da_quantity = e_load_true + e_ees - pv_true - e_fc  # >0 buying; <0 selling surplus electricity
        # node constraint
        if da_quantity < -self.mg_node_max:
            e_dif = -da_quantity - self.mg_node_max
            da_quantity = -self.mg_node_max
        if da_quantity > self.mg_node_max:
            e_dif = da_quantity - self.mg_node_max
            da_quantity = self.mg_node_max

        h_dif = np.abs(h_fc + h_gb - h_load_true - h_tes)  # heat balance difference (kwh)
        reward -= (g_fc + g_gb) * gas_price  # buy natural gas
        reward -= e_dif * c_p_e + ((h_dif / 10) ** 2 + h_dif) * c_p_h  # penalty
        # reward -= e_dif * c_p_e + h_dif * c_p_h  # penalty
        reward -= co2_emission * alpha_co2  # carbon tax

        # for da market, reward is without carbon tax and da market reward
        res_da_info = {
            "name": "res",
            "da_price": action[4], "tou_price": tou_price, "fit_price": fit_price,
            "da_quantity": da_quantity, "reward": reward
        }

        # ------------------- return market info and next state ------------------------ #

        self.step += 1  # next step
        s2_normalised_pv = self.normalized_states_info[self.step, 0]
        s2_normalised_e_load = self.normalized_states_info[self.step, 1]
        s2_normalised_h_load = self.normalized_states_info[self.step, 2]
        s2_normalised_tou_price = self.normalized_states_info[self.step, 3]

        next_state = np.hstack((self.S_ees, self.S_tes, s2_normalised_pv,
                                s2_normalised_e_load, s2_normalised_h_load, s2_normalised_tou_price))

        return res_da_info, next_state  # for debugging, return needed in-scope variable


class CommercialMicrogrid:
    def __init__(self, e_ees, p_ees, e_tes, p_tes, e_ehp_max, gas_gb_max, states_info, mg_node_max=1e6):

        self.n_states = 6  # no. of states
        self.n_actions = 5  # no. of actions (ehp, gb, ees, hss, da price)
        # all action max is 1

        # system variable
        self.E_ees_max = e_ees  # EES capacity
        self.P_ees_max = p_ees  # EES power capacity
        self.E_tes_max = e_tes  # TES capacity
        self.P_tes_max = p_tes  # TES power capacity
        self.e_ehp_max = e_ehp_max  # EHP input capacity (kwh)
        self.gas_gb_max = gas_gb_max  # GB input capacity (m3)
        self.mg_node_max = mg_node_max  # the max quantity can be submitted into the da market

        # internal states
        self.S_ees = 0.1  # SOC of EES
        self.S_tes = 0.1  # thermal storage level
        # external states dataframe
        self.states_info, self.normalized_states_info = initialize_external_states(states_info)
        self.fit_price = 0.04
        self.gas_price = 0.0338  # $/kwh
        # keep track of step for loading external states
        self.step = 0  # time step

    def reset(self):
        self.S_ees = 0.1
        self.S_tes = 0.1
        normalised_pv = self.normalized_states_info[0, 0]
        normalised_e_load = self.normalized_states_info[0, 1]
        normalised_h_load = self.normalized_states_info[0, 2]
        normalised_tou_price = self.normalized_states_info[0, 3]
        initial_states = np.hstack((self.S_ees, self.S_tes, normalised_pv,
                                    normalised_e_load, normalised_h_load, normalised_tou_price))  # reset state
        self.step = 0  # reset step
        return initial_states

    def sample(self):
        # random select actions [action_ehp, action_gb, action_ees, action_hss, action_da_price]
        actions = np.random.uniform(-1, 1, self.n_actions)
        return actions

    def env_step(self, action):
        # ----------------------in scope parameters--------------------#

        k_ehp = 2  # electric heat pump efficiency
        k_gb = 0.8  # gas boiler efficiency

        # wc_ees = 0.0091  # battery wear cost ($/kwh)
        k_ees = 0.95  # EeS charging/discharging efficiency
        k_tes = 0.9  # TES charging/discharging efficiency

        beta_gas = 0.245  # carbon intensity kg/kwh
        alpha_co2 = 0.0316  # carbon tax $/kg 0.0316
        reward = 0
        e_dif = 0

        # -----------------------load external states------------------#
        pv_true = self.states_info.iloc[self.step, 0]
        e_load_true = self.states_info.iloc[self.step, 1]
        h_load_true = self.states_info.iloc[self.step, 2]
        tou_price = self.states_info.iloc[self.step, 3]
        fit_price = self.fit_price
        gas_price = self.gas_price
        c_p_e = 3 * tou_price  # electricity penalty coefficient
        c_p_h = 3 * gas_price  # heat penalty coefficient

        # -----------------------load actions------------------#
        e_ehp = (action[0]+1) * 0.5 * self.e_ehp_max  # EHP input (kwh)
        h_ehp = e_ehp * k_ehp

        g_gb = (action[1]+1) * 0.5 * self.gas_gb_max  # GB input (m3)
        h_gb = g_gb * k_gb

        e_ees = action[2] * self.P_ees_max  # EES input electricity (kwh) >0 charge otherwise discharge
        h_tes = action[3] * self.P_tes_max  # TES input heat (kwh) >0 charge otherwise discharge

        # ---------------------calculate next internal state----------------#
        # EES
        if e_ees < 0:
            self.S_ees += (e_ees / self.E_ees_max) / k_ees
            if self.S_ees < 0.1:
                e_ees += (0.1 - self.S_ees) * self.E_ees_max * k_ees
                self.S_ees = 0.1
        else:
            self.S_ees += (e_ees / self.E_ees_max) * k_ees
            if self.S_ees > 1:
                e_ees -= (self.S_ees - 1) * self.E_ees_max / k_ees
                self.S_ees = 1
        # TES
        if h_tes < 0:
            self.S_tes += (h_tes / self.E_tes_max) / k_tes
            if self.S_tes < 0.1:
                h_tes += (0.1 - self.S_tes) * self.E_tes_max * k_tes
                self.S_tes = 0.1
        else:
            self.S_tes += (h_tes / self.E_tes_max) * k_tes
            if self.S_tes > 1:
                h_tes -= (self.S_tes - 1) * self.E_tes_max / k_tes
                self.S_tes = 1

        # -----------------calculate da market info and partial reward----------------#
        co2_emission = g_gb * beta_gas
        da_quantity = e_load_true + e_ees + e_ehp - pv_true  # >0 buying; <0 selling surplus electricity
        # node constraint
        if da_quantity < -self.mg_node_max:
            e_dif = -da_quantity - self.mg_node_max
            da_quantity = -self.mg_node_max
        if da_quantity > self.mg_node_max:
            e_dif = da_quantity - self.mg_node_max
            da_quantity = self.mg_node_max

        h_dif = np.abs(h_ehp + h_gb - h_load_true - h_tes)  # heat balance difference (m3)
        reward -= g_gb * gas_price  # buy natural gas
        reward -= e_dif * c_p_e + ((h_dif / 10) ** 2 + h_dif) * c_p_h  # penalty
        # reward -= e_dif * c_p_e + h_dif * c_p_h  # penalty
        reward -= co2_emission * alpha_co2  # carbon tax

        # for da market, reward is without carbon tax and da market reward
        com_da_info = {
            "name": "com",
            "da_price": action[4], "tou_price": tou_price, "fit_price": fit_price,
            "da_quantity": da_quantity, "reward": reward
        }

        # -------------------return market info and next state------------------------#

        self.step += 1  # next step
        s2_normalised_pv = self.normalized_states_info[self.step, 0]
        s2_normalised_e_load = self.normalized_states_info[self.step, 1]
        s2_normalised_h_load = self.normalized_states_info[self.step, 2]
        s2_normalised_tou_price = self.normalized_states_info[self.step, 3]

        next_state = np.hstack((self.S_ees, self.S_tes, s2_normalised_pv,
                                s2_normalised_e_load, s2_normalised_h_load, s2_normalised_tou_price))

        return com_da_info, next_state  # for debugging, return needed in-scope variable


class IndustrialMicrogrid:
    def __init__(self, e_ees, p_ees, e_tes, p_tes, gas_chp_max, gas_gb_max, states_info, mg_node_max=1e6):

        self.n_states = 6  # no. of states
        self.n_actions = 5  # no. of actions (chp, gb, ees, hss, da price)
        # all action max is 1

        # system variable
        self.E_ees_max = e_ees  # EES capacity
        self.P_ees_max = p_ees  # EES power capacity
        self.E_tes_max = e_tes  # TES capacity
        self.P_tes_max = p_tes  # TES power capacity
        self.gas_chp_max = gas_chp_max  # CHP input capacity (m3)
        self.gas_gb_max = gas_gb_max  # GB input capacity (m3)
        self.mg_node_max = mg_node_max  # the max quantity can be submitted into the da market

        # internal states
        self.S_ees = 0.1  # SOC of EES
        self.S_tes = 0.1  # thermal storage level
        # external states dataframe
        self.states_info, self.normalized_states_info = initialize_external_states(states_info)
        self.fit_price = 0.04
        self.gas_price = 0.0338  # $/kwh
        # keep track of step for loading external states
        self.step = 0  # time step

    def reset(self):
        self.S_ees = 0.1
        self.S_tes = 0.1
        normalised_pv = self.normalized_states_info[0, 0]
        normalised_e_load = self.normalized_states_info[0, 1]
        normalised_h_load = self.normalized_states_info[0, 2]
        normalised_tou_price = self.normalized_states_info[0, 3]
        initial_states = np.hstack((self.S_ees, self.S_tes, normalised_pv,
                                    normalised_e_load, normalised_h_load, normalised_tou_price))  # reset state
        self.step = 0  # reset step
        return initial_states

    def sample(self):
        # random select actions [action_chp, action_gb, action_ees, action_hss, action_da_price]
        actions = np.random.uniform(-1, 1, self.n_actions)
        return actions

    def env_step(self, action):
        # ----------------------in scope parameters--------------------#

        k_chp_e = 0.3  # fuel cell to electricity efficiency
        k_chp_h = 0.45  # fuel cell to heat efficiency
        k_gb = 0.8  # gas boiler efficiency

        # wc_ees = 0.0091  # battery wear cost ($/kwh)
        k_ees = 0.95  # EeS charging/discharging efficiency
        k_tes = 0.9  # TES charging/discharging efficiency

        beta_gas = 0.245  # carbon intensity kg/kwh
        alpha_co2 = 0.0316  # carbon tax $/kg 0.0316
        reward = 0
        e_dif = 0

        # -----------------------load external states------------------#
        pv_true = self.states_info.iloc[self.step, 0]
        e_load_true = self.states_info.iloc[self.step, 1]
        h_load_true = self.states_info.iloc[self.step, 2]
        tou_price = self.states_info.iloc[self.step, 3]
        fit_price = self.fit_price
        gas_price = self.gas_price
        c_p_e = 3 * tou_price  # electricity penalty coefficient
        c_p_h = 3 * gas_price  # heat penalty coefficient

        # -----------------------load actions------------------#
        g_chp = (action[0]+1) * 0.5 * self.gas_chp_max  # CHP input (m3)
        e_chp = g_chp * k_chp_e
        h_chp = g_chp * k_chp_h

        g_gb = (action[1]+1) * 0.5 * self.gas_gb_max  # GB input (m3)
        h_gb = g_gb * k_gb

        e_ees = action[2] * self.P_ees_max  # EES input electricity (kwh) >0 charge otherwise discharge
        h_tes = action[3] * self.P_tes_max  # TES input heat (kwh) >0 charge otherwise discharge

        # ---------------------calculate next internal state----------------#
        # EES
        if e_ees < 0:
            self.S_ees += (e_ees / self.E_ees_max) / k_ees
            if self.S_ees < 0.1:
                e_ees += (0.1 - self.S_ees) * self.E_ees_max * k_ees
                self.S_ees = 0.1
        else:
            self.S_ees += (e_ees / self.E_ees_max) * k_ees
            if self.S_ees > 1:
                e_ees -= (self.S_ees - 1) * self.E_ees_max / k_ees
                self.S_ees = 1
        # TES
        if h_tes < 0:
            self.S_tes += (h_tes / self.E_tes_max) / k_tes
            if self.S_tes < 0.1:
                h_tes += (0.1 - self.S_tes) * self.E_tes_max * k_tes
                self.S_tes = 0.1
        else:
            self.S_tes += (h_tes / self.E_tes_max) * k_tes
            if self.S_tes > 1:
                h_tes -= (self.S_tes - 1) * self.E_tes_max / k_tes
                self.S_tes = 1

        # -----------------calculate da market info and partial reward----------------#
        co2_emission = (g_chp + g_gb) * beta_gas
        da_quantity = e_load_true + e_ees - pv_true - e_chp  # >0 buying; <0 selling surplus electricity
        # node constraint
        if da_quantity < -self.mg_node_max:
            e_dif = -da_quantity - self.mg_node_max
            da_quantity = -self.mg_node_max
        if da_quantity > self.mg_node_max:
            e_dif = da_quantity - self.mg_node_max
            da_quantity = self.mg_node_max

        h_dif = np.abs(h_chp + h_gb - h_load_true - h_tes)  # heat balance difference (m3)
        reward -= (g_chp + g_gb) * gas_price  # buy natural gas
        reward -= e_dif * c_p_e + ((h_dif / 10) ** 2 + h_dif) * c_p_h  # penalty
        # reward -= e_dif * c_p_e + h_dif * c_p_h  # penalty
        reward -= co2_emission * alpha_co2  # carbon tax

        # for da market, reward is without carbon tax and da market reward
        ind_da_info = {
            "name": "ind",
            "da_price": action[4], "tou_price": tou_price, "fit_price": fit_price,
            "da_quantity": da_quantity, "reward": reward
        }

        # -------------------return market info and next state------------------------#

        self.step += 1  # next step
        s2_normalised_pv = self.normalized_states_info[self.step, 0]
        s2_normalised_e_load = self.normalized_states_info[self.step, 1]
        s2_normalised_h_load = self.normalized_states_info[self.step, 2]
        s2_normalised_tou_price = self.normalized_states_info[self.step, 3]

        next_state = np.hstack((self.S_ees, self.S_tes, s2_normalised_pv,
                                s2_normalised_e_load, s2_normalised_h_load, s2_normalised_tou_price))

        return ind_da_info, next_state
