"""Observation functions for traffic signals."""

from abc import abstractmethod

import numpy as np
from gymnasium import spaces

from .traffic_signal import TrafficSignal


class ObservationFunction:
    """Abstract base class for observation functions."""

    def __init__(self, ts: TrafficSignal):
        """Initialize observation function."""
        self.ts = ts

    @abstractmethod
    def __call__(self):
        """Subclasses must override this method."""
        pass

    @abstractmethod
    def observation_space(self):
        """Subclasses must override this method."""
        pass


class DefaultObservationFunction(ObservationFunction):
    """Default observation function for traffic signals."""

    def __init__(self, ts: TrafficSignal):
        """Initialize default observation function."""
        super().__init__(ts)

    def __call__(self) -> np.ndarray:
        """Return the default observation."""
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]  # one-hot encoding
        min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]
        density = self.ts.get_lanes_density()
        queue = self.ts.get_lanes_queue()
        veh_route_one_hot_distance = self.ts.get_veh_routeOneHot_Distance()

        # observation = np.array(phase_id + min_green + density + queue, dtype=np.float32)
        observation = np.array(phase_id + min_green + density + queue + veh_route_one_hot_distance, dtype=np.float32)

        # print("veh_route_one_hot_distance:", veh_route_one_hot_distance)

        return observation

    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
        sample_obs = self.__call__()  # ここで観測ベクトルを生成
        dim = sample_obs.shape[0]
        return spaces.Box(
            low=np.zeros(dim, dtype=np.float32),
            high=np.ones(dim, dtype=np.float32), 
        )
        # return spaces.Box(
        #     low=np.zeros(self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes), dtype=np.float32),
        #     high=np.ones(self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes), dtype=np.float32),
        # )