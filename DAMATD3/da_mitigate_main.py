import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from da_env_res import ResidentialMicrogrid
from da_env_ind import IndustrialMicrogrid
from da_env_com import CommercialMicrogrid
from td3_mitigate import TD3
import copy


def smooth(x, t):
    # last t average
    n = len(x)
    y = np.zeros(n)
    for i in range(n):
        start = max(0, i - (t-1))
        y[i] = float(sum(x[start:(i+1)])) / (i - start + 1)
    return y


def get_mitigate_action(s, noise_scale):
    a = td3_mitigation.mu_1h(np.array([s])).numpy()[0]
    a += np.random.randn(len(a)) * noise_scale
    return np.clip(a, -1, 1)


def test_agent():
    s_1h, _ = test_env.reset()
    rews = 0
    e_r = 0
    h_r = 0

    for k in range(24):

        att_s = test_env.state_attacked(attack_level, attack_clip)
        cor_a = td3_no_attack(np.array([s_1h])).numpy()[0]
        att_a = td3_no_attack(np.array([att_s])).numpy()[0]
        wr_et_cor_ec = np.concatenate((att_a[:4], cor_a[4:]))
        bat_pre = test_env.battery
        hydro_pre = test_env.hydrogen
        base_r, _, _, e_b, h_b = test_env.convert_energy(wr_et_cor_ec)
        test_env.step_1h -= 1  # above step is just for baseline, we don't want the time-step going forward
        test_env.battery = bat_pre
        test_env.hydrogen = hydro_pre

        # include wrong energy trading actions as states for mitigation model
        s_mit = np.concatenate((s_1h, att_a[:4]))
        mit_a_ec = get_mitigate_action(s_mit, 0)

        # Step the env
        mit_a_whole = np.concatenate((att_a[:4], mit_a_ec))
        r_mit, s2_1h, _, e_m, h_m = test_env.convert_energy(mit_a_whole)
        # real_mitigation_rewards += reward_mitigation

        # advantage reward for mitigation model training
        rews += r_mit - base_r
        e_r += e_m
        h_r += h_m

        s_1h = s2_1h

    return rews, e_r, h_r


def test_attack_mitigation():
    env_att = ResidentialMicrogrid(B_e=300, B_h2=30, pes_max=25, HT_p_max=3)
    env_WETCEC = ResidentialMicrogrid(B_e=300, B_h2=30, pes_max=25, HT_p_max=3)
    env_mit = ResidentialMicrogrid(B_e=300, B_h2=30, pes_max=25, HT_p_max=3)
    env_no_att = ResidentialMicrogrid(B_e=300, B_h2=30, pes_max=25, HT_p_max=3)
    s_att, _ = env_att.reset()
    s_WETCEC, _ = env_WETCEC.reset()
    s_mit, _ = env_mit.reset()
    s_no_att, _ = env_no_att.reset()
    rewards_att = []
    rewards_WETCEC = []
    rewards_mit = []
    rewards_no_att = []

    for k in range(24):

        # no attack model
        a_no_att = td3_no_attack(np.array([s_no_att])).numpy()[0]
        r_no_att, s2_no_att, _, _, _ = env_no_att.convert_energy(a_no_att)
        rewards_no_att.append(r_no_att)
        s_no_att = s2_no_att

        # same attack signal
        attack_signal = np.random.normal(0, attack_level)
        attack_signal = np.clip(attack_signal, -attack_clip, attack_clip)

        # attacked model
        s_att[1] += attack_signal
        a_att = td3_no_attack(np.array([s_att])).numpy()[0]
        r_att, s2_att, _, _, _ = env_att.convert_energy(a_att)
        rewards_att.append(r_att)
        s_att = s2_att

        # WETCEC model (wrong et correct ec model)
        a_WETCEC_C = td3_no_attack(np.array([s_WETCEC])).numpy()[0]
        s_WETCEC[1] += attack_signal
        a_WETCEC_W = td3_no_attack(np.array([s_WETCEC])).numpy()[0]
        a_WETCEC = np.concatenate((a_WETCEC_W[:4], a_WETCEC_C[4:]))
        r_WETCEC, s2_WETCEC, _, _, _ = env_WETCEC.convert_energy(a_WETCEC)
        rewards_WETCEC.append(r_WETCEC)
        s_WETCEC = s2_WETCEC

        # mitigation model
        s_mit_copy = copy.deepcopy(s_mit)
        s_mit[1] += attack_signal
        a_mit_w = td3_no_attack(np.array([s_mit])).numpy()[0]
        s_mit_et = np.concatenate((s_mit_copy, a_mit_w[:4]))
        a_mit_ec = get_mitigate_action(s_mit_et, 0)
        a_mit_whole = np.concatenate((a_mit_w[:4], a_mit_ec))
        r_mit, s2_mit, _, _, _ = env_mit.convert_energy(a_mit_whole)
        rewards_mit.append(r_mit)
        s_mit = s2_mit

    print(f'The daily reward for no attack model is: {np.sum(rewards_no_att)}')
    print(f'The daily reward for attacked model is: {np.sum(rewards_att)}')
    print(f'The daily reward for WETCEC model is: {np.sum(rewards_WETCEC)}')
    print(f'The daily reward for mitigation model is: {np.sum(rewards_mit)}')
    return rewards_att, rewards_WETCEC, rewards_mit, rewards_no_att


if __name__ == "__main__":
    tf.random.set_seed(42)
    np.random.seed(42)
    ss = 0
    env = ResidentialMicrogrid(B_e=300, B_h2=30, pes_max=25, HT_p_max=3)
    test_env = ResidentialMicrogrid(B_e=300, B_h2=30, pes_max=25, HT_p_max=3)

    num_train_episodes = 300
    test_agent_every = 20
    # start_steps = 5000
    # action_noise = 0.05
    action_noise_begin = 2
    action_noise_decay = 8e-4
    action_noise_final = 0.01
    action_noise = action_noise_begin
    policy_delay = 2
    attack_level = 1
    attack_clip = 2

    td3_args = {'q_lr_1h': 1e-3, 'mu_lr_1h': 1e-3, 'tau': 0.001, 'gamma': 0.995, 'batch_size': 100,
                'replay_capacity': int(1e5), 'target_noise': 0.2, 'target_noise_clip': 0.5}

    td3_no_attack = tf.keras.models.load_model("checkpoint/1h_model.h5")
    td3_mitigation = TD3(**td3_args)

    test_returns = []
    returns = []
    real_mitigation_returns = []
    q_losses_1h = []
    mu_losses_1h = []
    e_rewards = []
    h_rewards = []
    rewards_att_list = []
    rewards_WETCEC_list = []
    rewards_mit_list = []
    rewards_no_att_list = []

    for episode in range(num_train_episodes):
        t0 = datetime.now()
        state_1h, _ = env.reset()
        attacked_state = env.state_attacked(attack_level, attack_clip)

        rewards = []
        real_mitigation_rewards = 0
        if action_noise > action_noise_final:
            action_noise -= action_noise_decay

        for i in range(24):
            correct_actions = td3_no_attack(np.array([state_1h])).numpy()[0]
            attacked_actions = td3_no_attack(np.array([attacked_state])).numpy()[0]
            wrong_et_correct_ec = np.concatenate((attacked_actions[:4], correct_actions[4:]))

            battery_pre = env.battery
            hydrogen_pre = env.hydrogen
            baseline_reward, _, _, _, _ = env.convert_energy(wrong_et_correct_ec)
            # above step is just for baseline, we don't want the time-step going forward and storage to be changed
            env.step_1h -= 1
            env.battery = battery_pre
            env.hydrogen = hydrogen_pre


            # include wrong energy trading actions as states for mitigation model
            state_mitigation = np.concatenate((state_1h, attacked_actions[:4]))
            mitigate_action_ec = get_mitigate_action(state_mitigation, action_noise)

            # if ss > start_steps:
            #     actions = get_1h_action(state_1h, action_noise)
            # else:
            #     actions = env.sample_action()

            # Step the env
            mitigate_action_whole = np.concatenate((attacked_actions[:4], mitigate_action_ec))
            reward_mitigation, next_s_1h, _, _, _ = env.convert_energy(mitigate_action_whole)
            real_mitigation_rewards += reward_mitigation

            # advantage reward for mitigation model training
            reward = reward_mitigation - baseline_reward
            rewards.append(reward)

            # next mitigation state
            next_attacked_state = env.state_attacked(attack_level, attack_clip)
            next_attacked_actions = td3_no_attack(np.array([next_attacked_state])).numpy()[0]
            next_state_mitigation = np.concatenate((next_s_1h, next_attacked_actions[:4]))

            if i == 23:
                done = True
            else:
                done = False
            # at the end of 1 hour, store the 1h memory (including public books)
            td3_mitigation.memory.store(state_mitigation, mitigate_action_ec, reward, next_state_mitigation, done)
            state_1h = next_s_1h
            attacked_state = next_attacked_state

            # Keep track of the number of steps done
            ss += 1
            # if ss == start_steps:
            #     print("USING AGENT ACTIONS NOW")

        for j in range(24):

            q_loss_1h = td3_mitigation.train_critic_1h()
            q_losses_1h.append(q_loss_1h)
            td3_mitigation.soft_update(td3_mitigation.q_1h, td3_mitigation.t_q_1h)
            td3_mitigation.soft_update(td3_mitigation.q2_1h, td3_mitigation.t_q2_1h)

            # delayed policy update
            if j % policy_delay == 0:
                mu_loss_1h = td3_mitigation.train_actor_1h()
                mu_losses_1h.append(mu_loss_1h)
                td3_mitigation.soft_update(td3_mitigation.mu_1h, td3_mitigation.t_mu_1h)

        total_rewards = np.sum(rewards)
        test_rewards, test_e, test_h = test_agent()
        rewards_att, rewards_WETCEC, rewards_mit, rewards_no_att = test_attack_mitigation()

        rewards_att_list.append(np.sum(rewards_att))
        rewards_WETCEC_list.append(np.sum(rewards_WETCEC))
        rewards_mit_list.append(np.sum(rewards_mit))
        rewards_no_att_list.append(np.sum(rewards_no_att))
        returns.append(total_rewards)
        e_rewards.append(test_e)
        h_rewards.append(test_h)
        real_mitigation_returns.append(real_mitigation_rewards)
        test_returns.append(test_rewards)
        dt = datetime.now() - t0

        if episode % 20 == 0:
            print("Episode:", episode + 1, "Episode Return:", total_rewards, "Test Return:", test_rewards,
                  "one_epi Duration:", dt)

        # if episode > 0 and episode % test_agent_every == 0:
        #     test_agent(test_env)

    # # save the drl model
    # td3.mu_1h.save("1h_model.h5")
    
    
    plt.plot(returns, alpha=0.2, c='b',label='1')
    plt.plot(smooth(returns, 500), c='b',label='2')
    plt.title("Train returns")
    plt.legend()
    plt.savefig('figure/train_return_mitigate.pdf', bbox_inches = 'tight')
    plt.show()
    plt.close()

    plt.plot(test_returns, alpha=0.2, c='b',label='1')
    plt.plot(smooth(test_returns, 50), c='b',label='2')
    plt.title("Test returns")
    plt.legend()
    plt.savefig('figure/test_return_mitigate.pdf', bbox_inches = 'tight')
    plt.show()
    plt.close()
    
    plt.plot(e_rewards, alpha=0.2, c='b',label='1')
    plt.plot(smooth(e_rewards, 500), c='b',label='2')
    plt.title("Electricity reward")
    plt.legend()
    plt.savefig('figure/e_reward_mitigate.pdf', bbox_inches = 'tight')
    plt.show()
    plt.close()
    
    plt.plot(h_rewards, alpha=0.2, c='b',label='1')
    plt.plot(smooth(h_rewards, 500), c='b',label='2')
    plt.title("Heat reward")
    plt.legend()
    plt.savefig('figure/h_reward_mitigate.pdf', bbox_inches = 'tight')
    plt.show()
    plt.close()
    
    plt.plot(q_losses_1h, alpha=0.2, c='b',label='1')
    plt.plot(smooth(q_losses_1h, 5000), c='b',label='2')
    plt.title("q losses 1h")
    plt.legend()
    plt.savefig('figure/q_loss_mitigate.pdf', bbox_inches = 'tight')
    plt.show()
    plt.close()

    plt.plot(mu_losses_1h, alpha=0.2, c='b',label='1')
    plt.plot(smooth(mu_losses_1h, 5000), c='b',label='2')
    plt.title("mu losses 1h")
    plt.legend()
    plt.savefig('figure/mu_loss_mitigate.pdf', bbox_inches = 'tight')
    plt.show()
    plt.close()

    plt.plot(rewards_no_att_list, alpha=0.2, c='b',label='1')
    plt.plot(smooth(rewards_no_att_list, 5000), c='b',label='2')
    plt.title("daily reward for no attack model")
    plt.legend()
    plt.savefig('figure/reward_no_attack.pdf', bbox_inches = 'tight')
    plt.show()
    plt.close() 
   
    plt.plot(rewards_att_list, alpha=0.2, c='b',label='1')
    plt.plot(smooth(rewards_att_list, 5000), c='b',label='2')
    plt.title("daily reward for attacked model")
    plt.legend()
    plt.savefig('figure/reward_attack.pdf', bbox_inches = 'tight')
    plt.show()
    plt.close() 
   
    plt.plot(rewards_WETCEC_list, alpha=0.2, c='b',label='1')
    plt.plot(smooth(rewards_WETCEC_list, 5000), c='b',label='2')
    plt.title("daily reward for WETCEC model")
    plt.legend()
    plt.savefig('figure/reward_WETCEC.pdf', bbox_inches = 'tight')
    plt.show()
    plt.close() 
    
    plt.plot(rewards_mit_list, alpha=0.2, c='b',label='1')
    plt.plot(smooth(rewards_mit_list, 5000), c='b',label='2')
    plt.title("daily reward for mitigation model")
    plt.legend()
    plt.savefig('figure/reward_mitigation.pdf', bbox_inches = 'tight')
    plt.show()
    plt.close() 