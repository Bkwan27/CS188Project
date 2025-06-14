# CS188 Project

## Environment Setup

We recommend using a dedicated Anaconda or Python virtual environment for managing dependencies.

To get started:

```bash
conda create -n robosuite-env python=3.8 -y
conda activate robosuite-env
pip install -r requirements.txt
```

This will install all necessary packages, including robosuite and reinforcement learning libraries.

**NOTE: If you are not on a Mac, run all files using `python` instead of `mjpython`.**

## Project Structure and Code
### Soft Actor-Critic (SAC)
#### Training
To train a model, run `train.py` with the following flags:
- `--task`: the robosuite task you trained your model on and want to evaluate
  - Choices: lift, door
    - Default: lift
- `--continue_train`: if you want to continue training a model from its checkpoint
- `--reward_shaping`: if you want to use dense instead of sparse rewards for training
  - Default: True
- `--no_reward_shaping`: if you want to use sparse rewards
- `--checkpoint`: the path of the saved model `.zip` file you want to continue training
  - E.g. `"SAC_lift_dense_1mil.zip"`
- `--timesteps`: the number of timesteps to train your model
  - Default: 500,000

Now run:
```
mjpython main.py
```

#### Evaluation
To test your model after training, run `eval.py` with the following flags:
- `--task`: the robosuite task you trained your model on and want to evaluate
  - Choices: lift, door
    - Default: lift
- `--checkpoint`: the path of the model you want to evaluate
  - Required: True
    - E.g. `"demos/SAC_lift_dense_1mil.zip"`
- `--model`: the type of model you're evaluating
  - Choices: SAC, PPO
    - Default: SAC

Now run:
```
mjpython eval.py
```

**You may also run our demo models:**
SAC: lift, sparse, 1 million iterations
```
mjpython eval.py --task lift --checkpoint demos/SAC_lift_sparse_1mil.zip --model SAC
```
SAC: lift, dense, 1 million iterations
```
mjpython eval.py --task lift --checkpoint demos/SAC_lift_dense_1mil.zip --model SAC
```
PPO: lift, sparse, 10 million iterations
```
mjpython eval.py --task lift --checkpoint demos/PPO_lift_sparse_10mil.zip --model PPO
```
PPO: lift, dense, 10 million iterations
```
mjpython eval.py --task lift --checkpoint demos/PPO_lift_dense_10mil.zip --model PPO
```
SAC: door, dense, 1 million iterations
```
mjpython eval.py --task door --checkpoint demos/SAC_door_dense_1mil.zip --model SAC
```
PPO: door, dense, 3 million iterations
```
mjpython eval.py --task door --checkpoint demos/PPO_door_dense_3mil.zip --model PPO
```

### Proximal Policy Optimization (PPO)
The core PPO implementation, custom reward functions, and evaluation scripts are located in the `PPO` branch.

* **Dense Reward Function**: Defined in `env.py` under the `PPO` branch.
* **Lift Task Training**: See `main.py` for PPO training on the Lift environment.
* **Door Task Training**: See `door_main.py` for PPO training on the Door environment.
* **Evaluation**: Use `eval.py` to evaluate a trained PPO model.

To evaluate your own trained model:

1. Replace the file path in `model.load(...)` inside `eval.py` with the path to your `.zip` model.
2. Run the script:

   ```bash
   mjpython eval.py
   ```

## Notes

Make sure you are in the correct branch (e.g. `PPO`) to access all relevant code for training and evaluation.
