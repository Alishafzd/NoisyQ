from __future__ import annotations

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import *
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv

from random import randint
from math import sqrt, ceil


class SimpleEnv(MiniGridEnv):
    def __init__(
        self,
        size=20,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        max_steps: int | None = None,
        **kwargs,
    ):
        # Set the initial parameters
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            agent_view_size=1, 
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "grand mission"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)
        
        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        
                   
        ###################################### Place Terminals ######################################
        goals = []
        goal_num_prob = randint(5, 10)
        goal_num = ceil((width * height) ** 0.5 * goal_num_prob / 100)
        goal_counter = 1
        
        # Randomly place the walls
        while goal_counter <= goal_num:
            x = randint(1, width-1)
            y = randint(1, height-1)
            
            # A probability to place the goal further from the start pose
            dist_prob = round(sqrt(x ** 2 + y ** 2) / sqrt(width ** 2 + height ** 2) * 0.85) * 100
            
            if not(type(self.grid.get(x, y)).__name__ in ["Goal", "Wall"] or [x, y] == [0, 0]) and randint(0, 100) < dist_prob:
                
                goal_counter += 1
                goals.append([x, y])
                self.put_obj(Goal(), x, y)
                
        
        ###################################### Generate Walls ######################################
        
        wall_prob = randint(30, 35)/100  # Probablity to define the numbeer of walls
        wall_num =  (width-2) * (height-2) * wall_prob
        
        fill_prob = 15  # Probablity to set a cell as wall
        neighboring_prob = 5  # Fill probability for the first cell based on the neghboring cells status
        
        num = 0  # Wall num counter
        while num < wall_num:
            # First wall cell
            x = randint(0, width-2)
            y = randint(0, height-2)
            
            # Check the neighboring (if there is a neighboring wall, the probablity to select this cell as 
            # wall and continue is "neighboring_prob")
            if (type(self.grid.get(x+1, y)).__name__ in ["Goal", "Wall"] or type(self.grid.get(x, y+1)).__name__ in ["Goal", "Wall"] or \
                type(self.grid.get(x-1, y)).__name__ in ["Goal", "Wall"] or type(self.grid.get(x, y-1)).__name__ in ["Goal", "Wall"] or \
                type(self.grid.get(x+1, y+1)).__name__ in ["Goal", "Wall"] or type(self.grid.get(x-1, y-1)).__name__ in ["Goal", "Wall"]):
                
                resume_prob = neighboring_prob
                
            else:
                resume_prob = 100
            
            if randint(1, 100) < resume_prob:
                # Select the direction to extend the wall
                dir = randint(1, 4)
                
                while True:
                    if not([x, y] == [1, 1] or type(self.grid.get(x, y)).__name__ in ["Goal", "Wall"]):
                        self.grid.set(x, y, Wall())
                        
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
        self.has_path([1, 1])  # Solve the table
        
        # Reset, if path does not exist between one of the terminals and the start pose
        result = True
        for goal in goals:
            x = goal[0]
            y = goal[1]

            result = self.paths[x][y] and result
        
        if not result:
            self.reset()
        
        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "grand mission"
           
    
    def has_path(self, point):
        x_point = point[0]
        y_point = point[1]
        
        if not(self.paths[x_point+1][y_point] or type(self.grid.get(x_point+1, y_point)).__name__ == "Wall"):
            self.paths[x_point+1][y_point] = True
            self.has_path([x_point+1, y_point])
                
        if not(self.paths[x_point][y_point+1] or type(self.grid.get(x_point, y_point+1)).__name__ == "Wall"):
            self.paths[x_point][y_point+1] = True
            self.has_path([x_point, y_point+1])
                
        if not(self.paths[x_point-1][y_point] or type(self.grid.get(x_point-1, y_point)).__name__ == "Wall"):
            self.paths[x_point-1][y_point] = True
            self.has_path([x_point-1, y_point])
                
        if not(self.paths[x_point][y_point-1] or type(self.grid.get(x_point, y_point-1)).__name__ == "Wall"):
            self.paths[x_point][y_point-1] = True
            self.has_path([x_point, y_point-1])

        return 0


def main():
    env = SimpleEnv(render_mode="human")

    # enable manual control for testing
    manual_control = ManualControl(env, seed=42)
    manual_control.start()

    
if __name__ == "__main__":
    main()