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

**NOTE: If you are on a Mac, run all files using `mjpython` instead of `python`.**

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
python main.py
```

#### Evaluation
To test your model after training, run `eval.py` with the following flags:
- `--task`: the robosuite task you trained your model on and want to evaluate
  - Choices: lift, door
    - Default: lift
- `--checkpoint`: the path of the model you want to evaluate
  - Required: True
    - E.g. `"demos/SAC_lift_dense_1mil.zip"`

Now run:
```
python eval.py
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
   python eval.py
   ```

## Notes

Make sure you are in the correct branch (e.g. `PPO`) to access all relevant code for training and evaluation.
