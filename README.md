Sure! Here's a cleaner and more professional version of your README:

````markdown
# CS188 Project

## Environment Setup

We recommend using a dedicated Anaconda or Python virtual environment for managing dependencies.

To get started:

```bash
conda create -n robosuite-env python=3.8 -y
conda activate robosuite-env
pip install -r requirements.txt
````

This will install all necessary packages, including robosuite and reinforcement learning libraries.

## Project Structure and Code

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

Make sure you are in the correct branch (`PPO`) to access all relevant code for training and evaluation.

```

Let me know if you want to add usage examples, results visuals, or a section on experiments!
```

