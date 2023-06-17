import numpy as np
import random
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario

class Scenario(BaseScenario):
    def make_world(self,ep_pos=None):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_good_agents = 2
        num_adversaries = 2
        num_agents = num_adversaries + num_good_agents
        num_landmarks = 2

        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.size = 0.05
            agent.collide = True
            agent.silent = True
            if i < num_adversaries:
                agent.adversary = True
            else:
                agent.adversary = False
            agent.landmark_reached = 0
    
        world.agents[0].accel = 4.0
        world.agents[num_adversaries-1].accel = 4.0
        world.agents[num_adversaries].accel = 4.0
        world.agents[num_agents-1].accel = 4.0

        world.agents[0].max_speed = 4.0
        world.agents[1].max_speed = 4.0
        world.agents[2].max_speed = 4.0
        world.agents[3].max_speed = 4.0
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        world.first_land=False
        self.ep_pos=ep_pos
        self.pos_no=0
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.color = 0
        # make initial conditions
        self.reset_world(world)

        return world

    def negate(self):
        return 1 if random.random() < 0.5 else -1

    def get_positions(self, world, winning_team):
        pos0 = np.random.uniform(-1.5, +1.5, world.dim_p)
        pos1 = np.random.uniform(-1.5, +1.5, world.dim_p)
        pos2 = np.random.uniform(-1.5, +1.5, world.dim_p)
        pos3 = np.random.uniform(-1.5, +1.5, world.dim_p)
        return pos0, pos1, pos2, pos3

    def reset_world(self, world):
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = 0#np.array([0.1, 0.1, 0.1])
            # landmark.color[i + 1] += 0.8
            landmark.index = i

        # set random initial states
        for agent in world.agents:
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            agent.landmark_reached=0

        # fix the landmarks at two oppositie ends
        world.first_land=False
        if self.ep_pos:
            pos = self.ep_pos[(self.pos_no)%1000]
            self.pos_no += 1
            world.agents[0].state.p_pos = pos[0]
            world.agents[1].state.p_pos = pos[1]
            world.agents[2].state.p_pos = pos[2]
            world.agents[3].state.p_pos = pos[3]
        
            world.landmarks[0].state.p_pos = pos[4]
            world.landmarks[1].state.p_pos = pos[5]
            world.landmarks[0].state.p_vel = np.zeros(world.dim_p)
            world.landmarks[1].state.p_vel = np.zeros(world.dim_p)

        else:
            pos0, pos1, pos2, pos3 = self.get_positions(world, np.random.randint(0,2))
            world.agents[0].state.p_pos = pos0
            world.agents[1].state.p_pos = pos1
            world.agents[2].state.p_pos = pos2
            world.agents[3].state.p_pos = pos3
        
            world.landmarks[0].state.p_pos = np.random.uniform(-1.0, +1.0, world.dim_p)
            world.landmarks[1].state.p_pos = np.random.uniform(-1.0, +1.0, world.dim_p)
            world.landmarks[0].state.p_vel = np.zeros(world.dim_p)
            world.landmarks[1].state.p_vel = np.zeros(world.dim_p)
        
        

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        return self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    # whether two agents are colliding or not
    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def post_step(self, world):
        def dist(agent,landmark):
            return np.sqrt(np.sum(np.square(agent.state.p_pos - landmark.state.p_pos))) 
        if True:#not world.first_land:
            for agent in world.agents:
                if agent.landmark_reached==0:
                    for landmark in world.landmarks:
                        if landmark.color==0:
                            if dist(agent,landmark) <= 0.1:
                                if not world.first_land:
                                    world.first_land=True
                                landmark.color=1
                                agent.landmark_reached=1
    


    def agent_reward(self, agent, world):
        # the minimum distance to the goal
        stat=0
        if agent.landmark_reached==1:
            reward=0
        else:
            agent_dist = [-np.sqrt(np.sum(np.square(agent.state.p_pos - l.state.p_pos))) if l.color==0 else -float('inf') \
            for l in world.landmarks]
            reward = max(agent_dist)
            if reward==-float('inf'):
                reward=-0.1


        adversaries = self.adversaries(world)
        for a in adversaries:
            if self.is_collision(a, agent):
                reward -= 5

        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)

        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p]/1.5)
            reward -= bound(x)
 
        # calculate whether any team has reached the flag or not
        own_team = [-np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos)))\
         if l.color==0 and a.landmark_reached==0 else -float('inf')\
          for l in world.landmarks for a in world.agents if not a.adversary]
        opp_team = [-np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) \
        if l.color==0 and a.landmark_reached==0 else -float('inf') \
        for l in world.landmarks for a in world.agents if a.adversary]
        
        if not world.first_land:
            reward_val=30
        else:
            reward_val=5

        if np.max(own_team)>=-0.1:
            reward += reward_val
            ind=np.argmax(own_team)
            if ind in [0,2] and agent.name=='agent 2':
                stat=1
            if ind in [1,3] and agent.name=='agent 3':
                stat=1

        if np.max(opp_team)>=-0.1:
            reward -= reward_val

        return (reward,stat)


    def adversary_reward(self, agent, world):
        # minimum distance from the landmarks
        stat=0
        if agent.landmark_reached==1:
            reward=0
        else:
            agent_dist = [-np.sqrt(np.sum(np.square(agent.state.p_pos - l.state.p_pos))) if l.color==0 else -float('inf') \
            for l in world.landmarks]
            reward=max(agent_dist)

            if reward==-float('inf'):
                reward=-0.1

        good_agents = self.good_agents(world)
        for a in good_agents:
            if self.is_collision(a, agent):
                reward -= 5

        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)
        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p]/1.5)
            reward -= bound(x)
        
        
        # calculate whether any team has reached the flag or not
        own_team = [-np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos)))\
         if l.color==0 and a.landmark_reached==0 else -float('inf') \
         for l in world.landmarks for a in world.agents if a.adversary]
        opp_team = [-np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) \
        if l.color==0 and a.landmark_reached==0 else -float('inf') \
        for l in world.landmarks for a in world.agents if not a.adversary]
        
        
        if not world.first_land:
            reward_val=30
        else:
            reward_val=5

        if np.max(own_team)>=-0.1:
            reward += reward_val
            ind=np.argmax(own_team)
            if ind in [0,2] and agent.name=='agent 0':
                stat=1
            if ind in [1,3] and agent.name=='agent 1':
                stat=1

        if np.max(opp_team)>=-0.1:
            reward -= reward_val

        return (reward,stat)

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        
        # communication of all other agents
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent:
                continue
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            other_vel.append(other.state.p_vel)

        obs = np.concatenate([agent.state.p_pos] + entity_pos + other_pos + [agent.state.p_vel] + other_vel)
        return obs

    def done(self, agent, world):
        for a in world.agents:
            for l in world.landmarks:
                if (np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) <= 0.1):
                    a.landmark_reached += 1
                    return True 
        return False
