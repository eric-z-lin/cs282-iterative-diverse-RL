import argparse


def get_params():
    parser = argparse.ArgumentParser(
        description="Variable parameters based on the configuration of the machine or user's choice")

    parser.add_argument("--env_name", default="BipedalWalker-v3", type=str, help="Name of the environment.")
    parser.add_argument("--interval", default=20, type=int,
                        help="The interval specifies how often different parameters should be saved and printed,"
                             " counted by episodes.")
    parser.add_argument("--do_train", action="store_true",
                        help="The flag determines whether to train the agent or play with it.")
    parser.add_argument("--train_from_scratch", action="store_false",
                        help="The flag determines whether to train from scratch or continue previous tries.")
    parser.add_argument("--mem_size", default=int(1e+6), type=int, help="The memory size.")
    parser.add_argument("--n_skills", default=50, type=int, help="The number of skills to learn.")
    parser.add_argument("--n_skills_start", default=2, type=int, help="The number of skills to start learning with.")
    parser.add_argument("--reward_scale", default=1, type=float, help="The reward scaling factor introduced in SAC.")
    parser.add_argument("--seed", default=123, type=int,
                        help="The randomness' seed for torch, numpy, random & gym[env].")
    parser.add_argument("--max_n_episodes", default=500, type=int, help="The number of training episodes.")
    parser.add_argument("--max_episode_len", default=1000, type=int, help="The maximum length per episode during training.")
    parser.add_argument("--verbose", default=True, type=bool, help="If true, print statements every 10 episodes.")
    parser.add_argument("--skill_increment", default=0, type=int, help="The increment which skills is increased by each time.")
    parser.add_argument("--min_eps_before_inc", default=0, type=int, help="This is the number of rounds minimum before increasing skill.")
    parser.add_argument("--min_eps_before_inc_mult", default=1, type=float, help="Multiplier of previous flag.")

    parser.add_argument("--approach", default="none", type=str, help="Name of diversity increment approach {none, naive, reward, diverse1, diverse2}.")

    # Naive1
    parser.add_argument("--epsilon_diverse1_threshold", default=0.05, type=float, help="For diverse1, skills will be increased if percent change in moving average is less than epsilon.")
    parser.add_argument("--moving_avg_length_diverse1", default=5, type=int, help="The number of past episodes to use while calculating moving average for diverse1.")
    parser.add_argument("--epsilon_diverse2_threshold", default=0.05, type=float, help="Analogous to diverse1.")
    parser.add_argument("--moving_avg_length_diverse2", default=5, type=int, help="Analogous to diverse1.")

    parser_params = parser.parse_args()

    #  Parameters based on the DIAYN and SAC papers.
    # region default parameters
    default_params = {"lr": 3e-4,
                      "batch_size": 256,
                      "gamma": 0.99,
                      "alpha": 0.1,
                      "tau": 0.005,
                      "n_hiddens": 300
                      }
    # endregion
    total_params = {**vars(parser_params), **default_params}
    return total_params
