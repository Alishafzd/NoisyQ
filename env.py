from __future__ import annotations
from enum import IntEnum
from typing import Any, Iterable, SupportsFloat, TypeVar

from minigrid.core.constants import COLOR_NAMES, TILE_PIXELS
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import *
from minigrid.core.actions import Actions
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv
from gymnasium import spaces
from gymnasium.core import ActType, ObsType

from random import randint
from math import sqrt, ceil
import numpy as np


class CustomActions(IntEnum):
    # custom actions: left, right, up, down
    left = 0
    right = 1
    up = 2
    down = 3
    
    
# Monkey patching the key_handler for custom moves
def key_handler(self, event):
    key: str = event.key
    # print("pressed", key)

    if key == "escape":
        self.env.close()
        return
    if key == "backspace":
        self.reset()
        return

    key_to_action = {
        "left": CustomActions.left,
        "right": CustomActions.right,
        "up": CustomActions.up,
        "down": CustomActions.down,
    }
    if key in key_to_action.keys():
        action = key_to_action[key]
        self.step(action)
    # else:
        # print(key)
        
        
ManualControl.key_handler = key_handler
        
            
class SimpleEnv(MiniGridEnv):
    def __init__(
        self,
        mission_space: MissionSpace = "grand mission",
        size = 20,
        agent_start_pos = (1, 1),
        agent_start_dir = 0,
        max_steps: int | None = None,
        grid_size: int | None = None,
        width: int | None = None,
        height: int | None = None,
        see_through_walls: bool = False,
        agent_view_size: int = 1,
        render_mode: str = "human",
        screen_size: int | None = 640,
        highlight: bool = True,
        tile_size: int = TILE_PIXELS,
        agent_pov: bool = False,
        display = False,
        **kwargs,
    ):
        mission_space = MissionSpace(mission_func=self._gen_mission)
        self.step_count = 0

        if max_steps is None:
            max_steps = 4 * size**2

        width = size
        height = size 
        # Set the initial parameters
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.reward = 0
        
        # Generate walls and terminals
        self._place_objs(size, size)

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size**2

        # Initialize mission
        self.mission = mission_space.sample()

        # Can't set both grid_size and width/height
        if grid_size:
            assert width is None and height is None
            width = grid_size
            height = grid_size
        assert width is not None and height is not None

        # Action enumeration for this environment
        self.actions = CustomActions

        # Actions are discrete integer values
        self.action_space = spaces.Discrete(len(self.actions))

        # Number of cells (width and height) in the agent view
        assert agent_view_size % 2 == 1
        assert agent_view_size >= 0
        self.agent_view_size = agent_view_size

        # Observations are dictionaries containing an
        # encoding of the grid and a textual 'mission' string
        image_observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.agent_view_size, self.agent_view_size, 3),
            dtype="uint8",
        )
        self.observation_space = spaces.Dict(
            {
                "image": image_observation_space,
                "direction": spaces.Discrete(4),
                "mission": mission_space,
            }
        )

        # Range of possible rewards
        self.reward_range = (0, 1)

        # Environment configuration
        self.width = width
        self.height = height

        assert isinstance(
            max_steps, int
        ), f"The argument max_steps must be an integer, got: {type(max_steps)}"
        self.max_steps = 1000000

        self.see_through_walls = False

        # Current position and direction of the agent
        self.agent_pos: np.ndarray | tuple[int, int] = None
        self.agent_dir: int = None

        # Rendering attributes 
        if display:
            self.screen_size = screen_size
            self.render_size = None
            self.window = None
            self.clock = None
            
            self.render_mode = render_mode
            self.highlight = highlight
            self.tile_size = tile_size
            self.agent_pov = agent_pov

    @staticmethod
    def _gen_mission():
        return "grand mission"
      
      
    def step(
        self, action: int
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self.step_count += 1
        
        current_pos = self.agent_pos
        x = current_pos[0]
        y = current_pos[1]

        reward = self.reward - 1
        terminated = False
        truncated = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Rotate left
        if action == self.actions.left.value:
            fwd_pos = (x-1, y)
            
            # Get the contents of the cell in front of the agent
            fwd_cell = self.grid.get(*fwd_pos)
            
            if fwd_cell is None or fwd_cell.can_overlap():
                self.agent_pos = tuple(fwd_pos)
            if fwd_cell is not None and fwd_cell.type == "goal":
                terminated = True
            if fwd_cell is not None and fwd_cell.type == "lava":
                terminated = True

        # Rotate right
        elif action == self.actions.right.value:
            fwd_pos = (x+1, y)
            fwd_cell = self.grid.get(*fwd_pos)
            
            if fwd_cell is None or fwd_cell.can_overlap():
                self.agent_pos = tuple(fwd_pos)
            if fwd_cell is not None and fwd_cell.type == "goal":
                terminated = True
            if fwd_cell is not None and fwd_cell.type == "lava":
                terminated = True

        # Move up
        elif action == self.actions.up.value:
            fwd_pos = (x, y-1)
            fwd_cell = self.grid.get(*fwd_pos)
            
            if fwd_cell is None or fwd_cell.can_overlap():
                self.agent_pos = tuple(fwd_pos)
            if fwd_cell is not None and fwd_cell.type == "goal":
                terminated = True
            if fwd_cell is not None and fwd_cell.type == "lava":
                terminated = True

        # Move down
        elif action == self.actions.down.value:
            fwd_pos = (x, y+1)
            fwd_cell = self.grid.get(*fwd_pos)
            
            if fwd_cell is None or fwd_cell.can_overlap():
                self.agent_pos = tuple(fwd_pos)
            if fwd_cell is not None and fwd_cell.type == "goal":
                terminated = True
            if fwd_cell is not None and fwd_cell.type == "lava":
                terminated = True

        # Done action (not used by default)
        elif action == self.actions.done:
            pass

        else:
            raise ValueError(f"Unknown action: {action}")

        if self.step_count >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()

        obs = self.agent_pos
        self.reward = reward

        return obs, reward, terminated, truncated

    
    def _place_objs(self, width, height):
        """
        Randomly picks goals and walls cells and stores them in allocated parameters. Later, we use the cell coordinates found 
        here to generate the Walls and Goals as many times as the algorithm takes to finish
        """
        self.goals = []
        self.walls = []
        
        ###################################### Terminals ######################################
        goal_num_prob = randint(5, 10)
        goal_num = ceil((width * height) ** 0.5 * goal_num_prob / 100)
        goal_counter = 1
        
        # Randomly place the goals
        while goal_counter <= goal_num:
            x = randint(1, width-1)
            y = randint(1, height-1)
            point = np.array([x, y])
            
            # A probability to place the goal further from the start pose
            dist_prob = round(sqrt(x ** 2 + y ** 2) / sqrt(width ** 2 + height ** 2) * 0.85) * 100
            
            if not(x == 0 or x == width-1 or y == 0 or y == height-1 or [x,y] in self.goals) and randint(0, 100) < dist_prob:
                
                goal_counter += 1
                self.goals.append([x,y])
                
        ###################################### Walls ######################################
        
        wall_prob = randint(30, 35)/100  # Probablity to define the numbeer of walls
        wall_num =  (width-2) * (height-2) * wall_prob
        
        fill_prob = 15  # Probablity to set a cell as wall
        neighboring_prob = 5  # Fill probability for the first cell based on the neghboring cells status
        
        num = 0  # Wall num counter
        dist_walls = 1
        while num < wall_num:
            # First wall cell
            x = randint(0, width-2)
            y = randint(0, height-2)
            point = np.array([x, y])
            
            # Check the neighboring (if there is a neighboring wall, the probablity to select this cell as 
            # wall and continue is "neighboring_prob")
            # If the suggested cell is already a wall or goal, it won't be picked
            dist_goals = np.linalg.norm(np.array(self.goals) - point, axis=1)
            if len(self.walls) >= 1:
                dist_walls = np.linalg.norm(np.array(self.walls) - point, axis=1)
                
            if (np.min(dist_goals) < 1.42 or np.min(dist_walls) < 1.42):  
                resume_prob = neighboring_prob
            elif ([x,y] in self.walls or [x,y] in self.goals):
                resume_prob = -1
            else:
                resume_prob = 100
            
            if randint(1, 100) < resume_prob:
                # Select the direction to extend the wall
                dir = randint(1, 4)
                
                while True:                    
                    # Check coincidence 
                    if not([x, y] == [1, 1] or x == 0 or x == width-1 or y == 0 or y == height-1 or [x,y] in self.walls or [x,y] in self.goals):
                        self.walls.append([x,y])
                        
                        num += 1
                        
                        fill = randint(0, 100)
                        if fill > fill_prob:
                            if dir == 1: x = x + 1
                            elif dir == 2: y = y + 1
                            elif dir == 3: x = x - 1
                            elif dir == 4: y = y - 1
                        else:
                            break
                    else:
                        break
               
        # "paths" table defines if for each cell there is a path between start pose and that cell
        self.paths = [[False for _ in range(height)] for _ in range(width)]
        self.paths[1][1] = True  
        self.has_path([1, 1], width, height)  # Solve the table
        

        # Redo, if path does not exist between one of the terminals and the start pose
        result = True
        for goal in self.goals:
            x = int(goal[0])
            y = int(goal[1])
            
            result = self.paths[x][y] and result
        
        if not result:
            self._place_objs(width, height)
    

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)
        
        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        
        for wall in self.walls:
            x = int(wall[0])
            y = int(wall[1])

            self.put_obj(Wall(), x, y)
            
        for goal in self.goals:
            x = int(goal[0])
            y = int(goal[1])

            self.put_obj(Goal(), x, y)
        
        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "grand mission"
        
    def reset(self):
        super().reset()
        
        return self.agent_pos
           
    
    def has_path(self, point, width, height):
        """
        Fills out a table of the size of the environment. The table shows for each cell if there is a path from the cell to the start point.
        """
        x = point[0]
        y = point[1]
        
        if (x == 0 or x == width-1 or y == 0 or y == height-1):
            return 0
        
        else:
            if not(self.paths[x+1][y] or [x+1, y] in self.walls):
                self.paths[x+1][y] = True
                self.has_path([x+1, y], width, height)
                    
            if not(self.paths[x][y+1] or [x, y+1] in self.walls):
                self.paths[x][y+1] = True
                self.has_path([x, y+1], width, height)
                    
            if not(self.paths[x-1][y] or [x-1, y] in self.walls):
                self.paths[x-1][y] = True
                self.has_path([x-1, y], width, height)
                    
            if not(self.paths[x][y-1] or [x, y-1] in self.walls):
                self.paths[x][y-1] = True
                self.has_path([x, y-1], width, height)

        return 0
    
    
# def main():
#     env = SimpleEnv(display=True)
#     print(env.reset())
#     while True:
#         action = env.action_space.sample()
#         new_obs, rew, done, _ = env.step(action)

    # enable manual control for testing
    # manual_control = ManualControl(env, seed=42)
    # manual_control.start()

    
# if __name__ == "__main__":
#     main()