# Traffic-Signal-Control-DRC

This repository contains the source code for the paper:
**"Deep Reinforcement Learning for Traffic Signal Control Utilizing Autonomous Vehicle Route Information"**, accepted at **ICCAE 2026**.

## Requirements
* Python 3.x
* SUMO (Simulation of Urban MObility)
* gymnasium>=0.28
* pettingzoo>=1.24.3
* numpy
* pandas
* pillow
* sumolib>=1.14.0
* traci>=1.14.0
* sumo-rl

## Usage
1.  Clone this repository.
2.  Run the training script:
    ```bash
    python ./experiments/DRC_train.py
    ```
3.  Run the test script:
    ```bash
    python ./experiments/test.py --model-path [model name]
    ```    

## Files
* `DRC_train.py`: Training code for DRC. By modifying the net file settings, training can be performed in various environments.
* `test.py`: Test code for the trained agent. By default, the metrics used are accumulative_waiting_time_per_vehicles, total_arrived_vehicles, and co2_emissions_per_vehicles.

## References
This project relies on the **SUMO-RL** library. We would like to thank the authors for their contribution to the community.
If you use the SUMO-RL environment in your work, please cite their repository as follows:

```bibtex
@misc{sumorl,
    author = {Lucas N. Alegre},
    title = {{SUMO-RL}},
    year = {2019},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{[https://github.com/LucasAlegre/sumo-rl](https://github.com/LucasAlegre/sumo-rl)}},
}
