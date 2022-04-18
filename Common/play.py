# from mujoco_py.generated import const
from mujoco_py import GlfwContext
import cv2
import numpy as np
import os
import imageio

GlfwContext(offscreen=True)


class Play:
    def __init__(self, env, agent, n_skills):
        self.env = env
        self.agent = agent
        self.n_skills = n_skills
        self.agent.set_policy_net_to_cpu_mode()
        self.agent.set_policy_net_to_eval_mode()
        if not os.path.exists("Vid/"):
            os.mkdir("Vid/")

    @staticmethod
    def concat_state_latent(s, z_, n):
        z_one_hot = np.zeros(n)
        z_one_hot[z_] = 1
        return np.concatenate([s, z_one_hot])

    def evaluate(self):

        for z in range(self.n_skills):
            images = []
            s = self.env.reset()
            s = self.concat_state_latent(s, z, self.n_skills)
            episode_reward = 0
            for _ in range(self.env.spec.max_episode_steps):
                action = self.agent.choose_action(s)
                s_, r, done, _ = self.env.step(action)
                s_ = self.concat_state_latent(s_, z, self.n_skills)
                episode_reward += r
                if done:
                    break
                s = s_
                I = self.env.render(mode='rgb_array')
                I = cv2.cvtColor(I, cv2.COLOR_RGB2BGR)
                I = cv2.resize(I, (250, 250))
                images.append(I)
            print(f"skill: {z}, episode reward:{episode_reward:.1f}")
            imageio.mimsave(f"Vid/{self.env.unwrapped.spec.id}_skill{z}" + ".gif", images)
        self.env.close()
        cv2.destroyAllWindows()
