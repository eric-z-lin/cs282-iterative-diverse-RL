import gym
from Brain import SACAgent
from Common import Play, Logger, get_params
import numpy as np
from tqdm import tqdm
import mujoco_py


def concat_state_latent(s, z_, n):
    z_one_hot = np.zeros(n)
    z_one_hot[z_] = 1
    return np.concatenate([s, z_one_hot])


if __name__ == "__main__":
    params = get_params()

    test_env = gym.make(params["env_name"])
    n_states = test_env.observation_space.shape[0]
    n_actions = test_env.action_space.shape[0]
    action_bounds = [test_env.action_space.low[0], test_env.action_space.high[0]]

    params.update({"n_states": n_states,
                   "n_actions": n_actions,
                   "action_bounds": action_bounds})
    print("params:", params)
    test_env.close()
    del test_env, n_states, n_actions, action_bounds

    env = gym.make(params["env_name"])

    p_z = np.full(params["n_skills"], 1 / params["n_skills"])
    agent = SACAgent(p_z=p_z, **params)
    logger = Logger(agent, **params)

    if params["do_train"]:

        if not params["train_from_scratch"]:    # load from logger
            episode, last_logq_zs, np_rng_state, *env_rng_states, torch_rng_state, random_rng_state = logger.load_weights()
            agent.hard_update_target_network()
            min_episode = episode
            np.random.set_state(np_rng_state)
            env.np_random.set_state(env_rng_states[0])
            env.observation_space.np_random.set_state(env_rng_states[1])
            env.action_space.np_random.set_state(env_rng_states[2])
            agent.set_rng_states(torch_rng_state, random_rng_state)
            print("Keep training from previous run.")

        else:
            min_episode = 0
            last_logq_zs = 0
            np.random.seed(params["seed"])
            env.seed(params["seed"])
            env.observation_space.seed(params["seed"])
            env.action_space.seed(params["seed"])
            print("Training from scratch.")

        logger.on()
        
        curr_num_skills = params["n_skills_start"]
        curr_num_skills = min(curr_num_skills, params["n_skills"])
        print(f'curr_num_skills {curr_num_skills}')
        p_z = np.full(curr_num_skills, 1 / curr_num_skills)
        agent.p_z = np.tile(p_z, agent.batch_size).reshape(agent.batch_size, curr_num_skills)
        
        max_reward = -np.inf
        max_reward_ep = 0
        last_increment_ep = 0

        diversity_rewards_lst = []
        diversity_actiondiff_lst = []

        for episode in tqdm(range(1 + min_episode, params["max_n_episodes"] + 1)):

            # z = np.random.choice(params["n_skills"], p=p_z)
            selected_approach = params["approach"].lower()
            if selected_approach != "none" and (episode - last_increment_ep) >= params["min_eps_before_inc"]:
                increment = False
                if params["approach"] == "naive":   # Naive approach
                    if (episode+1) % params["interval"] == 0:    # Skills += K every N episodes
                        increment = True
                elif params["approach"] == "reward":      # Reward stagnant approach
                    if episode - max_reward_ep >= params["min_eps_before_inc"]:
                        increment = True
                elif params["approach"] == "diverse1":  # Diverse1 approach
                    if len(diversity_rewards_lst) > (2*params["moving_avg_length_diverse1"]):
                        moving_avg_1 = sum(diversity_rewards_lst[-2*params["moving_avg_length_diverse1"]:-params["moving_avg_length_diverse1"]]) / params["moving_avg_length_diverse1"]
                        moving_avg_2 = sum(diversity_rewards_lst[-params["moving_avg_length_diverse1"]:]) / params["moving_avg_length_diverse1"]
                        
                        if moving_avg_1 != 0:
                            perc_change = (moving_avg_2 - moving_avg_1) / moving_avg_1
                        if perc_change > params["epsilon_diverse1_threshold"]:
                            increment = True
                elif params["approach"] == "diverse2": 
                    if len(diversity_actiondiff_lst) > (2*params["moving_avg_length_diverse2"]): 
                        moving_avg_1 = sum(diversity_actiondiff_lst[-2*params["moving_avg_length_diverse2"]:-params["moving_avg_length_diverse2"]]) / params["moving_avg_length_diverse2"]
                        moving_avg_2 = sum(diversity_actiondiff_lst[-params["moving_avg_length_diverse2"]:]) / params["moving_avg_length_diverse2"]
                        
                        if moving_avg_1 != 0:
                            perc_change = (moving_avg_2 - moving_avg_1) / moving_avg_1
                            if perc_change > params["epsilon_diverse2_threshold"]:
                                increment = True
                else:
                    raise ValueError("Not valid value for selected approach.")
                
                if increment:     
                    last_increment_ep = episode
                    params["min_eps_before_inc"] *= params["min_eps_before_inc_mult"]       # increase episodes in between skill increases
                    diversity_rewards_lst = []        # reset rewards list when skill is added
                    diversity_actiondiff_lst = []

                    curr_num_skills += params["skill_increment"]
                    curr_num_skills = min(curr_num_skills, params["n_skills"])
                    print(f'curr_num_skills {curr_num_skills}')
                    p_z = np.full(curr_num_skills, 1 / curr_num_skills)
                    agent.p_z = np.tile(p_z, agent.batch_size).reshape(agent.batch_size, curr_num_skills)
            z = np.random.choice(curr_num_skills, p=p_z)

            state = env.reset()
            env_state = state
            state = concat_state_latent(state, z, params["n_skills"])
            episode_reward = 0
            diverse2_curr_ep_total = 0
            episode_steps = 0
            diverse1_cumulative_reward = 0
            total_steps = 0
            logq_zses = []

            max_n_steps = min(params["max_episode_len"], env.spec.max_episode_steps)
            for step in range(1, 1 + max_n_steps):

                if params["approach"] == "diverse2":
                    for skill1 in range(curr_num_skills - params["skill_increment"], curr_num_skills): 
                        # skill1 = curr_num_skills -1 
                        state_latent1 = concat_state_latent(env_state, skill1, params["n_skills"])
                        action1 = agent.choose_action(state_latent1)
                        for skill2 in range(curr_num_skills): 
                            state_latent2 = concat_state_latent(env_state, skill2, params["n_skills"])
                            action2 = agent.choose_action(state_latent2)
                            diverse2_curr_ep_total += np.linalg.norm(skill1-skill2)

                action = agent.choose_action(state)
                next_state, reward, done, _ = env.step(action)
                env_state = next_state
                next_state = concat_state_latent(next_state, z, params["n_skills"])
                agent.store(state, z, done, action, next_state)
                train_res = agent.train()
                if train_res is None:
                    logq_zses.append(last_logq_zs)
                else:
                    logq_zs, diversity_rewards = train_res
                    logq_zses.append(logq_zs)
                    diverse1_cumulative_reward += sum(diversity_rewards)
                episode_reward += reward
                state = next_state
                total_steps = step
                if done:
                    break

            # Append episode reward to list
            diversity_rewards_lst.append(diverse1_cumulative_reward)
            diversity_actiondiff_lst.append(diverse2_curr_ep_total / total_steps)

            # Update max reward
            if episode_reward > max_reward:
                max_reward = episode_reward
                max_reward_ep = episode

            avg_logqzs = sum(logq_zses) / len(logq_zses)
            logger.log(episode,
                       episode_reward,
                       z,
                       curr_num_skills,
                       avg_logqzs,
                       step,
                       diverse1_cumulative_reward,
                       diverse2_curr_ep_total,
                       np.random.get_state(),
                       env.np_random.get_state(),
                       env.observation_space.np_random.get_state(),
                       env.action_space.np_random.get_state(),
                       *agent.get_rng_states(),
                       )
            if (episode+1) % 20 == 0 and params["verbose"]:
                print(f'Episode {episode}, reward {episode_reward}, z {z}, avg_logqzs {avg_logqzs}')

    else:
        logger.load_weights()
        player = Play(env, agent, n_skills=params["n_skills"])
        player.evaluate()
