# cs282-iterative-diverse-RL
Authors: Eric Lin and Catherine Zeng  


### Objective
We extend the original DIAYN paper by proposing four separate approaches to iteratively learn the number of skills. We evaluate this on `Hopper` and `BipedalWalker` environments.

### Usage
- **If you want to run default experiment
```shell
python3 main.py --mem_size=1000000 --env_name="Hopper-v3" --interval=20 --do_train --n_skills=15 --n_skills_start=15 --max_n_episodes=300
```
- **If you want to visualize data
```shell
tensorboard --logdir='/your/path/here'
```
- **If you want to create gifs, duplicate the saved params.pth into the main folder of the env (e.g. Hopper/) and run without the do_train flag
