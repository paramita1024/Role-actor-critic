import numpy as np
import seaborn as sns
from multiagent.core import World, Agent, Landmark, Wall
from multiagent.scenario import BaseScenario

class Scenario(BaseScenario):
    def make_world(self, ep_length=50,ep_pos=None):
        world = World()
        # set world properties
        world.cache_dists = False
        world.dim_c = 2
        world.dim_p = 2
        num_agents = 4
        num_adversaries = 2
        num_resources = 10
        
        self.ep_length = ep_length
        self.t = 1
        self.ep_pos=ep_pos

        type_of_resources = 3
        world.resource_types = list(range(type_of_resources))

        # producers are agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.i = i
            agent.name = 'agent %d' % i
            agent.adversary = True if i < num_adversaries else False

            agent.collide = False
            agent.silent = True
            agent.ghost = True
            agent.holding = np.array([0.0, 0.0, 0.0])

            agent.reward = 0
            agent.size = 0.05
            agent.accel = 1.5
            agent.initial_mass = 1.0
            agent.max_speed = 1.0

        world.landmarks = [Landmark() for i in range(num_resources)]
        for i, landmark in enumerate(world.landmarks):
            landmark.i = i + num_agents
            landmark.name = 'penny %d' % i
            landmark.type = world.resource_types[i%type_of_resources]

            landmark.alive = True
            landmark.collide = False
            landmark.movable = False

            landmark.size = 0.05
            landmark.boundary = False

        world.walls = []

        # initial conditions
        self.reset_world(world)
        return world

    def team_2(self, world):
        return [a for a in world.agents if not a.adversary]

    def team_1(self, world):
        return [a for a in world.agents if a.adversary]

    def resources(self, world):
        return [r for r in world.landmarks]

    def is_collision(self, entity1, entity2, world):
        dist = np.sqrt(np.sum(np.square(entity1.state.p_pos - entity2.state.p_pos)))
        return True if dist < entity1.size + entity2.size else False

    def reset_world(self, world):
        
        self.t = 1
        num_resources = len(world.resource_types)

        for i, agent in enumerate(world.agents):
            agent.state.p_pos = np.random.uniform(low=-1, high=1, size=world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            agent.holding = np.array([0.0, 0.0, 0.0])
            agent.reward = 0

        for i, landmark in enumerate(world.landmarks):
            bound = 0.95
            landmark.type = world.resource_types[i%num_resources]
            landmark.state.p_pos = np.random.uniform(low=-bound, high=bound, size=world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
            landmark.alive = True

    def post_step(self, world):

        self.t += 1
        bound = 0.95
        for l in self.resources(world):
            for a in world.agents:
                if self.is_collision(l, a, world):
                    a.holding[l.type] += 1
                    l.state.p_pos = np.random.uniform(low=-bound, high=bound, size=world.dim_p)

    def calculate_reward(self, world):
        team_1 = np.array([0.0, 0.0, 0.0])
        team_2 = np.array([0.0, 0.0, 0.0])

        for a in self.team_1(world):
            team_1 = team_1 + a.holding
        team_1 = team_1/(np.sqrt(np.sum(np.square(team_1)))) if np.sum(np.square(team_1))!=0 else team_1

        for a in self.team_2(world):
            team_2 = team_2 + a.holding
        team_2 = team_2/(np.sqrt(np.sum(np.square(team_2)))) if np.sum(np.square(team_2))!=0 else team_2

        M = np.array([
                [0.0, -1.0, 1.0],
                [1.0, 0.0, -1.0],
                [-1.0, 1.0, 0.0]
            ])

        reward = 100*np.matmul(np.matmul(team_1, M), team_2)
        return reward


    def team_1_reward(self, agent, world):
        reward = 0
        curr_hold=np.array([0,0,0])
        for r in self.resources(world):
            if self.is_collision(agent, r, world):
                reward += 0.5
                curr_hold[r.type] +=1
            else:
                reward -= 0.1
        agent.reward += reward

        def bound(x):
            if x<1.0:
                return 0
            if x<1.1:
                return (x-1.0)*10
            return min(np.exp(2*(x-1.1)), 10)

        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            reward -= bound(x/1.5)

        if self.t == self.ep_length:
            reward += self.calculate_reward(world)

        return (reward,curr.hold)

    def team_2_reward(self, agent, world):
        reward = 0
        curr_hold=np.array([0,0,0])
        for r in self.resources(world):
            if self.is_collision(agent, r, world):
                reward += 0.5
                curr_hold[r.type] +=1
            else:
                reward -= 0.1
        agent.reward += reward

        def bound(x):
            if x<1.0:
                return 0
            if x<1.1:
                return (x-1.0)*10
            return min(np.exp(2*(x-1.1)), 10)

        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            reward -= bound(x/1.5)

        if self.t == self.ep_length:
            reward -= self.calculate_reward(world)

        return (reward,curr_hold)

    def reward(self, agent, world):
        return self.team_1_reward(agent, world) if agent.adversary else self.team_2_reward(agent, world)

    def agent_encoding(self, agent, world):
        encoding = []
        encoding.append(agent.state.p_vel)
        return np.concatenate(encoding)

    def resource_encoding(self, resource, world):
        encoding = []
        n_resource_types = len(world.resource_types)
        encoding.append((np.arange(n_resource_types)==resource.type))
        return np.concatenate(encoding)

    def observation(self, agent, world):
        obs = []
        obs.append(agent.state.p_pos)
        obs.append(agent.holding)
        obs.append(self.agent_encoding(agent, world))
        
        if agent.adversary:
            for a in self.team_1(world):
                if a is not agent:
                    obs.append(a.state.p_pos - agent.state.p_pos)
                    obs.append(self.agent_encoding(a, world))
            for a in self.team_2(world):
                obs.append(a.state.p_pos - agent.state.p_pos)
                obs.append(self.agent_encoding(a, world))
        else:
            for a in self.team_2(world):
                if a is not agent:
                    obs.append(a.state.p_pos - agent.state.p_pos)
                    obs.append(self.agent_encoding(a, world))
            for a in self.team_1(world):
                obs.append(a.state.p_pos - agent.state.p_pos)
                obs.append(self.agent_encoding(a, world))

        for r in self.resources(world):
            obs.append(r.state.p_pos - agent.state.p_pos)
            obs.append(self.resource_encoding(r, world))

        return np.concatenate(obs)
