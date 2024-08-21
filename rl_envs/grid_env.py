import gymnasium
import numpy as np

class SimpleGridEnv(gymnasium.Env):
    """
    Simple grid environment where the agent has to reach the goal.  

    The agent can move in 4 directions: up, down, left, right.

    Parameters
    -----------
    size: int
        The size of the grid. The grid will be size x size.
    """
    def __init__(self, size):

        self.size = size # The size of the grid.

        # Specs
        self.observation_space = gymnasium.spaces.Box(low=0, high=size-1, shape=(4,), dtype=np.int32) # (agent_x, agent_y, goal_x, goal_y)
        self.action_space = gymnasium.spaces.Discrete(4) # 0:up, 1:down, 2:left, 3:right

        # State
        self.goal_pos = (size-1, size-1)
        self.agent_pos = (0, 0)

        # Placeholder
        self.history = []

        self.reset()

    def reset(self):
        self.agent_pos = np.random.randint(0, self.size, size=2)
        self.goal_pos = np.random.randint(0, self.size, size=2)
        return self._get_obs()

    def step(self, action):

        self.history += [
            self.agent_pos.copy()
        ]

        if action == 0: # up
            self.agent_pos[1] = min(self.size - 1, self.agent_pos[1] + 1)
        elif action == 1: # down
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 2: # left
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 3: # right
            self.agent_pos[0] = min(self.size - 1, self.agent_pos[0] + 1)

        # Check if reached goal
        if self.agent_pos[0] == self.goal_pos[0] and self.agent_pos[1] == self.goal_pos[1]:
            reward = 10
            done = True
        else:
            reward = -0.1 # TODO: Check if negative reward is better
            done = False

        info = {
            "history": self.history
        }

        return self._get_obs(), reward, done, info


    def _get_obs(self):
        return np.array([self.agent_pos[0], self.agent_pos[1], self.goal_pos[0], self.goal_pos[1]])

    def render(self,/, pixel_size=1, with_history=False):
        # Generate grid with RGB, black for background, red for agent, green for goal

        # Create empty grid
        grid = np.zeros((self.size, self.size, 3))

        # If with_history is True, mark the history with darker color.
        if with_history:
            for pos in self.history:
                grid[pos[1], pos[0]] = [0, 0, 128]

        grid[self.agent_pos[1], self.agent_pos[0]] = [255, 0, 0] # Note that matrix is (y, x)
        grid[self.goal_pos[1], self.goal_pos[0]] = [0, 255, 0] # Note that matrix is (y, x)



        grid = np.flip(grid, axis=0) # Flip the grid along x-axis to make it match the coordinate system

        # Expand the grid to pixel size
        grid = np.repeat(np.repeat(grid, pixel_size, axis=0), pixel_size, axis=1)

        return grid.astype(np.uint8)