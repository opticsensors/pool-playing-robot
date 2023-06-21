"""Just randomly impulses balls"""

import math
import random

import gymnasium as gym
import numpy as np
import pygame
import pymunk
import pymunk.pygame_util
from gymnasium import spaces


class BilliardEnv_0(gym.Env):
    """360º rotation + impulse, 1 ball"""

    def __init__(self):
        # We define the action space
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0]),
            high=np.array([360.0, 1000.0]),
            dtype=np.float32,
        )

        # Physics
        self._setup_physics()

        # Target
        self._setup_target()

    def _setup_physics(self):
        # Step 0
        self.space = pymunk.Space()
        self.space.damping = 0.9

        # Step 1
        inertia = pymunk.moment_for_circle(mass=1, inner_radius=0, outer_radius=1)
        body = pymunk.Body(mass=0.1, moment=inertia)
        body.position = (0, 0)
        shape = pymunk.Circle(body, radius=20)
        self.space.add(body, shape)

    def _setup_target(self):
        angle = 2 * math.pi * random.random()
        radius = 200 + 400 * random.random()
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        self.target: tuple[int, int] = (int(x), int(y))

    def reset(self):
        super().reset()

    def reward_function(self):
        pass

    def step(self, action):
        # We hit the ball
        angle_in_radians = np.radians(action[0])
        force_x = action[1] * np.cos(angle_in_radians)
        force_y = action[1] * np.sin(angle_in_radians)
        self.space.bodies[0].apply_force_at_local_point((force_x, force_y))

        # We live in a simulation
        total_time = 60.0
        step_size = 1 / 60.0
        num_steps = int(total_time / step_size)
        for _ in range(num_steps):
            self.space.step(step_size)

        # Reward is the distance to the target
        ball_position = self.space.bodies[0].position
        distance_to_target = np.linalg.norm(
            np.array(ball_position) - np.array(self.target)
        )

        return None, -distance_to_target, True, {}

    def render(self):
        # Simple pygame draw example
        pygame.init()
        pygame.display.init()
        self.screen = pygame.display.set_mode((1000, 1000))
        pygame.display.set_caption("Billiard")

        # We draw!
        self.screen.fill((68, 77, 67))
        pygame.draw.circle(self.screen, (63, 43, 37), (500, 500), 20)
        pygame.draw.circle(
            self.screen, (50, 54, 66), (self.target[0] + 500, self.target[1] + 500), 20
        )
        for i in self.space.shapes:
            if type(i) == pymunk.shapes.Circle:
                pygame.draw.circle(
                    self.screen,
                    (163, 73, 67),
                    (int(i.body.position[0]) + 500, int(i.body.position[1]) + 500),
                    int(i.radius),
                )
        pygame.display.flip()

        # Show until I press 'esc'
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return


if __name__ == "__main__":
    env = BilliardEnv_0()

    for i in range(10):
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print(reward)

    print("Done!")
