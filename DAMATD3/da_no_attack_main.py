import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from da_env_res import ResidentialMicrogrid
from da_env_ind import IndustrialMicrogrid
from da_env_com import CommercialMicrogrid
from td3_da import TD3


def smooth(x, t):
    # last t average
    n = len(x)
    y = np.zeros(n)
    for i in range(n):
        start = max(0, i - (t-1))
        y[i] = float(sum(x[start:(i+1)])) / (i - start + 1)
    return y


def get_1h_action(s, noise_scale):
    a = td3.mu_1h(np.array([s])).numpy()[0]
    a += np.random.randn(len(a)) * noise_scale
    return np.clip(a, -1, 1)


def test_agent(environment):
    s_1h, _ = environment.reset()

    rews = 0
    e_r = 0
    h_r = 0
    for k in range(24):
        a = get_1h_action(s_1h, 0)

        # Step the env
        r, s2_1h, _, e, h = environment.convert_energy(a)
        
        rews += r
        e_r += e
        h_r += h
        s_1h = s2_1h

    return rews, e_r, h_r


if __name__ == "__main__":
    tf.random.set_seed(42)
    np.random.seed(42)
    ss = 0
    env = ResidentialMicrogrid(B_e=300, B_h2=30, pes_max=25, HT_p_max=3)
    test_env = ResidentialMicrogrid(B_e=300, B_h2=30, pes_max=25, HT_p_max=3)
#     env = CommercialMicrogrid(B_e=300, B_h2=30, pes_max=25, HT_p_max=3)
#     test_env = CommercialMicrogrid(B_e=300, B_h2=30, pes_max=25, HT_p_max=3)
#     env = IndustrialMicrogrid(B_e=300, B_h2=30, pes_max=25, HT_p_max=3)
#     test_env = IndustrialMicrogrid(B_e=300, B_h2=30, pes_max=25, HT_p_max=3)

    num_train_episodes = 300
    test_agent_every = 50
    # start_steps = 5000
    # action_noise = 0.05
    action_noise_begin = 1
    action_noise_decay = 4.75e-3
    action_noise_final = 0.05
    action_noise = action_noise_begin
    policy_delay = 2

    td3_args = {'n_states_1h': env.n_features_1h, 'n_actions_1h': env.n_actions, 'n_books': env.n_books_1h,
                'q_lr_1h': 5e-3, 'mu_lr_1h': 5e-3, 'tau': 0.001, 'gamma': 0.995, 'batch_size': 128,
                'replay_capacity': int(1e5), 'target_noise': 0.1, 'target_noise_clip': 0.25}

    td3 = TD3(**td3_args)

    test_returns = []
    returns = []
    q_losses_1h = []
    mu_losses_1h = []
    e_rewards = []
    h_rewards = []

    for episode in range(num_train_episodes):
        t0 = datetime.now()
        state_1h, k_books = env.reset()

        rewards = []

        if action_noise > action_noise_final:
            action_noise -= action_noise_decay

        for i in range(24):
            # [电价，电量，热价，热量，WE产能控制，FC产能控制]
            actions = get_1h_action(state_1h, action_noise)
            
            # if ss > start_steps:
            #     actions = get_1h_action(state_1h, action_noise)
            # else:
            #     actions = env.sample_action()

            # Step the env
            reward_1h, next_s_1h, next_k_books, _, _ = env.convert_energy(actions)
#             print('actions shape: ',actions.shape)
#             print('actions: ',actions)
            rewards.append(reward_1h)
        
            if i == 23:
                done = True
            else:
                done = False
            # at the end of 1 hour, store the 1h memory (including public books)
            td3.memory.store(state_1h, actions, k_books, reward_1h, next_s_1h, next_k_books, done)
            state_1h = next_s_1h
            k_books = next_k_books

            # Keep track of the number of steps done
            ss += 1
            # if ss == start_steps:
            #     print("USING AGENT ACTIONS NOW")

        for j in range(24):

            q_loss_1h = td3.train_critic_1h()
            q_losses_1h.append(q_loss_1h)
            td3.soft_update(td3.q_1h, td3.t_q_1h)
            td3.soft_update(td3.q2_1h, td3.t_q2_1h)

            # delayed policy update
            if j % policy_delay == 0:
                mu_loss_1h = td3.train_actor_1h()
                mu_losses_1h.append(mu_loss_1h)
                td3.soft_update(td3.mu_1h, td3.t_mu_1h)

        total_rewards = np.sum(rewards)
        test_rewards, test_e, test_h = test_agent(test_env)

        returns.append(total_rewards)
        test_returns.append(test_rewards)
        e_rewards.append(test_e)
        h_rewards.append(test_h)
#         print(test_e)
        dt = datetime.now() - t0

        if episode % 20 == 0:
            print("Episode:", episode + 1, "Episode Return:", total_rewards, "Test Return:", test_rewards,
                  "one_epi Duration:", dt)

        # if episode > 0 and episode % test_agent_every == 0:
        #     test_agent(test_env)

    # save the drl model
    td3.mu_1h.save("checkpoint/1h_model.h5")

    plt.plot(returns, alpha=0.2, c='b',label='1')
    plt.plot(smooth(returns, 500), c='b',label='2')
    plt.title("Train returns")
    plt.legend()
    plt.savefig('figure/train_return.pdf', bbox_inches = 'tight')
    plt.show()
    plt.close()
    
    plt.plot(e_rewards, alpha=0.2, c='b',label='1')
    plt.plot(smooth(e_rewards, 500), c='b',label='2')
    plt.title("Electricity reward")
    plt.legend()
    plt.savefig('figure/e_reward.pdf', bbox_inches = 'tight')
    plt.show()
    plt.close()
    
    plt.plot(h_rewards, alpha=0.2, c='b',label='1')
    plt.plot(smooth(h_rewards, 500), c='b',label='2')
    plt.title("Heat reward")
    plt.legend()
    plt.savefig('figure/h_reward.pdf', bbox_inches = 'tight')
    plt.show()
    plt.close()
    
    plt.plot(test_returns, alpha=0.2, c='b',label='1')
    plt.plot(smooth(test_returns, 50), c='b',label='2')
    plt.title("Test returns")
    plt.legend()
    plt.savefig('figure/test_return.pdf', bbox_inches = 'tight')
    plt.show()
    plt.close()
    #
    plt.plot(q_losses_1h, alpha=0.2, c='b',label='1')
    plt.plot(smooth(q_losses_1h, 5000), c='b',label='2')
    plt.title("q losses 1h")
    plt.legend()
    plt.savefig('figure/q_loss.pdf', bbox_inches = 'tight')
    plt.show()
    plt.close()
    #
    plt.plot(mu_losses_1h, alpha=0.2, c='b',label='1')
    plt.plot(smooth(mu_losses_1h, 5000), c='b',label='2')
    plt.title("mu losses 1h")
    plt.legend()
    plt.savefig('figure/mu_loss.pdf', bbox_inches = 'tight')
    plt.show()
    plt.close()