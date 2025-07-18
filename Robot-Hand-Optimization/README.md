# Robot Hand Analysis and Morphology Optimization

This is an implementation of the data-driven design optimization method presented in the paper [**Data-efficient Co-Adaptation of Morphology and Behaviour with Deep Reinforcement Learning**](https://research.fb.com/publications/data-efficient-co-adaptation-of-morphology-and-behaviour-with-deep-reinforcement-learning/).

## Installation

Make sure that PyTorch is installed. You find more information here: https://pytorch.org/

First, clone this repository to your local computer as usual.
Then, install the required packages via pip by executing `pip3 install -r requirements.txt`.

The current version uses the latest version of rlkit by [Vitchyr Pong](https://github.com/vitchyr/rlkit).

Clone the rlkit with
```bash
git clone https://github.com/vitchyr/rlkit.git
```
Now, set in your terminal the environment variable PYTHONPATH with
```bash
export PYTHONPATH=/path/to/rlkit/
```
where the folder `/path/to/rlkit` contains the folder `rlkit`. This enables us
to import rlkit with `import rlkit`.

You may have to set the environmental variable every time you open a new terminal.

## Starting experiments

After setting the environmental variable and installing the packages you can
proceed to run the experiments.
There are experimental configurations set up in `experiment_configs.py`.
You can execute them with
```bash
python3 main.py allegrohand
```

You may change the configs or add new ones. Make sure to add new configurations to
the `config_dict` in `experiment_configs.py`.

## Data logging
If you execute these commands, they will automatically create directories in which
the performance and achieved rewards will be stored. Each experiment creates
a specific folder with the current date/time and a random string as name.
You can find in this folder a copy of the config you executed and one csv file
for each design on which the reinforcement learning algorithm was executed.
Each csv file contains three rows: The type of the design (either 'Initial', 'Optimized' or 'Random');
The design vector; And the subsequent, cumulative rewards for each episode/trial.

## Citation

```
@inproceedings{luck2019coadapt,
  title={Data-efficient Co-Adaptation of Morphology and Behaviour with Deep Reinforcement Learning},
  author={Luck, Kevin Sebastian and Ben Amor, Heni and Calandra, Roberto},
  booktitle={Conference on Robot Learning},
  year={2019}
}
```

## Acknowledgements
This project would have been harder to implement without the great work of
the developers behind rlkit and pybullet.

The reinforcement learning loop makes extensive use of rlkit, a framework developed
and maintained by Vitchyr Pong. You find this repository [here](https://github.com/vitchyr/rlkit).

Tasks were simulated in [PyBullet](https://pybullet.org/wordpress/), the
repository can be found [here](https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet).