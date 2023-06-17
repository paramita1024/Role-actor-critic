import numpy as np
import seaborn as sns
from multiagent.core import World, Agent, Landmark, Wall
from multiagent.scenario import BaseScenario

class Scenario(BaseScenario):
    def make_world(self,ep_pos=None,speed_dict=None,\
        speed_file='~/speed.txt',speed_change_opt=True,\
        intrinsic_re=None,team_re_frac=0,ag_re_frac=0\
        ):
        world = World()
        # set world properties
        world.cache_dists = False
        world.dim_c = 2
        world.dim_p = 2
        num_agents = 4
        num_producers = num_agents
        num_adversaries = 2
        num_consumers = 2
        num_resources = 4

        world.resource_types = list(range(num_consumers))
        world.resource_colors = np.array(sns.color_palette(n_colors=num_consumers))

        # producers are agents
        world.agents = [Agent() for i in range(num_producers)]
        # self.speed_change_opt=speed_change_opt
        for i, agent in enumerate(world.agents):
            agent.i = i
            agent.name = 'agent %d' % i
            agent.adversary = True if i < num_adversaries else False
            agent.color = 0 # np.array([0.85, 0.85, 0.85])

            agent.collide = False
            agent.silent = True
            agent.ghost = True
            agent.holding = None

            agent.reward = 0
            agent.size = 0.05
            agent.initial_mass = 1.0 
            agent.accel = 1.5
            agent.max_speed = 1.0

            # agent.picked=[0.0 for i in range(num_consumers)]   
            # agent.dropped=[0.0 for i in range(num_consumers)]
            # if self.speed_change_opt:
            #     agent.max_speed = speed_dict['max_speed'][i]
            #     agent.accel =speed_dict['accel'][i]
            #     agent.init_speed=float(agent.max_speed)

        world.landmarks = [Landmark() for i in range(num_consumers + num_resources)]
        for i, landmark in enumerate(world.landmarks):
            landmark.i = i + num_agents
            landmark.name = 'treasure %d' % i
            landmark.consumer = True if i < num_consumers else False

            landmark.respawn_prob = 0
            landmark.type = world.resource_types[i%num_consumers]
            landmark.color = world.resource_colors[landmark.type]

            landmark.alive = True
            landmark.collide = False
            landmark.movable = False

            landmark.size = 0.05
            landmark.boundary = False

        world.walls = []
        self.ep_pos = ep_pos 
        # self.landmark_count=[]
        # self.intrinsic_re=intrinsic_re
        # self.team_re_frac=team_re_frac
        # self.ag_re_frac=ag_re_frac
        self.pos_no = 0
        # self.speed_file = speed_file
        # initial conditions
        self.reset_world(world)
        return world

    def producers(self, world):
        return world.agents

    def team_2(self, world):
        return [a for a in world.agents if not a.adversary]

    def team_1(self, world):
        return [a for a in world.agents if a.adversary]

    def consumers(self, world):
        return [c for c in world.landmarks if c.consumer]

    def resources(self, world):
        return [r for r in world.landmarks if not r.consumer]

    def is_collision(self, entity1, entity2, world):
        dist = np.sqrt(np.sum(np.square(entity1.state.p_pos - entity2.state.p_pos)))
        return True if dist < entity1.size + entity2.size else False

    def reset_world(self,world):
        
        # create ep pos and pos no 
        num_consumers = len(self.consumers(world))
        # if curr_status:
        # self.curr_status=curr_status
        if self.ep_pos:
            pos = self.ep_pos[(self.pos_no)%1000]
            self.pos_no += 1
            for i, agent in enumerate(world.agents):
                agent.state.p_pos = pos[i] #np.random.uniform(low=-1, high=1, size=world.dim_p)
                agent.state.p_vel = np.zeros(world.dim_p)
                agent.state.c = np.zeros(world.dim_c)
                agent.holding = None
                agent.color = 0 # np.array([0.85, 0.85, 0.85])
                agent.reward = 0

            for i, landmark in enumerate(world.landmarks):
                bound = 0.95
                landmark.type = world.resource_types[i%num_consumers]
                landmark.color = world.resource_colors[landmark.type]
                landmark.state.p_pos = pos[i+len(world.agents)]#np.random.uniform(low=-bound, high=bound, size=world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)
                landmark.alive = True
        else:
            for i, agent in enumerate(world.agents):
                agent.state.p_pos = np.random.uniform(low=-1, high=1, size=world.dim_p)
                agent.state.p_vel = np.zeros(world.dim_p)
                agent.state.c = np.zeros(world.dim_c)
                agent.holding = None
                agent.color = 0 # np.array([0.85, 0.85, 0.85])
                agent.reward = 0

            for i, landmark in enumerate(world.landmarks):
                bound = 0.95
                landmark.type = world.resource_types[i%num_consumers]
                landmark.color = world.resource_colors[landmark.type]
                landmark.state.p_pos = np.random.uniform(low=-bound, high=bound, size=world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)
                landmark.alive = True
        # if weak_vel:
        #     for i, agent in enumerate(world.agents):
        #         agent.max_speed=weak_vel[i]
        #         agent.accel=agent.max_speed
        # else:
        #     for i, agent in enumerate(world.agents):
        #         tmp=2*(agent.picked[0]+agent.dropped[0])+(agent.picked[1]+agent.dropped[1])
        #         agent.max_speed = agent.init_speed+\
        #             (4 - agent.init_speed)*(1-np.exp(-(tmp)/float(10000) ))
        #         agent.accel = agent.max_speed



    def post_step(self, world):

        for l in self.resources(world):
            if l.alive:
                for a in self.producers(world):
                    if a.holding is None and self.is_collision(l, a, world):
                        l.alive = False
                        a.holding = l.type
                        a.color = 1 # 0.85 * l.color
                        l.state.p_pos = np.array([-999.,-999.])
                        break

        for a in self.producers(world):
            if a.holding is not None:
                for c in self.consumers(world):
                    if c.alive and c.type == a.holding and self.is_collision(a, c, world):
                        c.alive = False
                        a.holding = None
                        a.color = 2 # np.array([0.85, 0.85, 0.85])
                        c.state.p_pos = np.array([999., 999.])

        
    # def episode_info(self,agent,world):

    #     pick=-1
    #     drop=-1
    #     for r in self.resources(world):
    #         if r.alive and a.holding is None and self.is_collision(a, r, world):
    #             pick=r.type
    #     for c in self.consumers(world):
    #         if c.alive and c.type==a.holding and self.is_collision(a, c, world):
    #             drop=c.type
    #     return [pick, drop]

    def team_1_reward(self, agent, world):

        reward = 0
        stat=[0,0]
        if agent.color == 0:
            reward = -1.0
        elif agent.color == 1:
            for c in self.consumers(world):
                if c.type==agent.holding:
                    if c.alive==True:
                        reward = -np.sqrt(np.sum(np.square(c.state.p_pos - agent.state.p_pos)))/2.0
                    else:
                        reward = -1.0
        reward_list_pick=[20,20]
        reward_list_drop=[40,40]
        for a in self.team_1(world):
            for r in self.resources(world):
                if r.alive and a.holding is None and self.is_collision(a, r, world):
                    reward+=reward_list_pick[r.type]
                    if agent.i==a.i:
                        stat[r.type]=1

                    # a.picked[r.type] += 0.5
            for c in self.consumers(world):
                if c.alive and c.type==a.holding and self.is_collision(a, c, world):
                    reward+=reward_list_drop[c.type]
                    if agent.i==a.i:
                        stat[c.type]=2

                    # a.dropped[c.type] += 0.5

        def bound(x):
            if x<1.0:
                return 0
            if x<1.1:
                return (x-1.0)*10
            return min(np.exp(2*(x-1.1)), 10)

        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            reward -= bound(x/1.5)

        agent.reward += reward
        return (reward,stat)

    def team_2_reward(self, agent, world):

        reward = 0
        stat=[0,0]
        if agent.color == 0:
            reward = -1.0
        elif agent.color == 1:
            for c in self.consumers(world):
                if c.type==agent.holding:
                    if c.alive==True:
                        reward = -np.sqrt(np.sum(np.square(c.state.p_pos - agent.state.p_pos)))/2.0
                    else:
                        reward = -1.0
        reward_list_pick=[20,20]
        reward_list_drop=[40,40]

        # def get_dynamic_re_frac(landmark_vector_init=None):
        #     # if not whether_speed:
        #     #     landmark_vector = np.average(np.array(landmark_vector_init), axis=0).flatten()
        #     max_landmark_vec =np.max(landmark_vector_init)#.max()
        #     if max_landmark_vec == 0 : 
        #         print('current landmark', landmark_vector_init)
        #         return 0,0
        #     landmark_vector = np.array([ float(i)/float(max_landmark_vec) for i in landmark_vector_init])
        #     # max_diff = max([ abs(i-j) for i in  landmark_vector for j in  landmark_vector])
        #     team_re_fr = max((np.average(landmark_vector[:2])-np.average(landmark_vector[2:])),0)
        #     ag_re_fr = max((landmark_vector[2] - landmark_vector[3]),0 )
        #     return team_re_fr, ag_re_fr
        
        for a in self.team_2(world):
            for r in self.resources(world):
                if r.alive and a.holding is None and self.is_collision(a, r, world):
                    reward+=reward_list_pick[r.type]
                    if agent.i==a.i:
                        stat[r.type]=1

                    # if self.intrinsic_re:
                    #     if 'Dynamic' in self.intrinsic_re:
                    #         if 'Speed' in self.intrinsic_re:
                    #             speed=[agent.max_speed for agent in world.agents()]
                    #             self.team_re_frac, self.ag_re_frac=get_dynamic_re_frac(landmark_vector_init=speed, whether_speed=True)
                    #             # print('Fractions')
                    #             # print(self.team_re_frac, self.ag_re_frac)
                    #         if 'Landmark' in self.intrinsic_re:
                    #             self.team_re_frac, self.ag_re_frac=get_dynamic_re_frac(landmark_vector_init=self.curr_status)                    
                    #             # print('Fractions')
                    #             # print(self.team_re_frac, self.ag_re_frac)
                    #     if self.intrinsic_re in ['StaticTeam','StaticAgent','DynamicLandmark','DynamicSpeed']:
                    #         reward+=self.team_re_frac*float(reward_list_pick[r.type])
                    #     if self.intrinsic_re in ['StaticAgent','DynamicLandmark','DynamicSpeed'] and a.i==3:
                    #         reward+=self.ag_re_frac*float(reward_list_pick[r.type])
                    # a.picked[r.type] += 0.5
            for c in self.consumers(world):
                if c.alive and c.type==a.holding and self.is_collision(a, c, world):
                    reward+=reward_list_drop[c.type]
                    if agent.i==a.i:
                        stat[c.type]=2

                    # if self.intrinsic_re:
                    #     if self.intrinsic_re in ['StaticTeam','StaticAgent']:
                    #         reward+=self.team_re_frac*float(reward_list_drop[c.type])
                    #     if self.intrinsic_re in ['StaticAgent'] and a.i==3:
                    #         reward+=self.ag_re_frac*float(reward_list_drop[c.type])
                    # a.dropped[c.type] += 0.5
            
        def bound(x):
            if x<1.0:
                return 0
            if x<1.1:
                return (x-1.0)*10
            return min(np.exp(2*(x-1.1)), 10)

        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            reward -= bound(x/1.5)

        agent.reward += reward

        return (reward,stat)

    def reward(self, agent, world):
        return self.team_1_reward(agent, world) if agent.adversary else self.team_2_reward(agent, world)

    def agent_encoding(self, agent, world):
        encoding = []
        n_resource_types = len(world.resource_types)
        encoding.append(agent.state.p_vel) # 2
        encoding.append((np.arange(n_resource_types)==agent.holding)) # 2
        return np.concatenate(encoding) # 4

    def consumer_encoding(self, consumer, world):
        encoding = []
        n_resource_types = len(world.resource_types)
        encoding.append((np.arange(n_resource_types)==consumer.type))
        encoding.append((np.array([True])==consumer.alive))
        encoding.append(np.array([1]))
        return np.concatenate(encoding)

    def resource_encoding(self, resource, world):
        encoding = []
        n_resource_types = len(world.resource_types)
        encoding.append((np.arange(n_resource_types)==resource.type)) # 2
        encoding.append((np.array([True])==resource.alive)) # 1
        encoding.append(np.array([0])) # 1
        return np.concatenate(encoding)

    def observation(self, agent, world):
        obs = []
        obs.append(agent.state.p_pos) # 0 1
        obs.append(self.agent_encoding(agent, world)) # 2 3 4 5
        if agent.adversary:
            for a in self.team_1(world):
                if a is not agent:
                    obs.append(a.state.p_pos - agent.state.p_pos) # 6 7
                    obs.append(self.agent_encoding(a, world)) # 8 9 10 11
            for a in self.team_2(world):
                obs.append(a.state.p_pos - agent.state.p_pos) # next 12
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
            if r.alive:
                obs.append(r.state.p_pos - agent.state.p_pos) # len 2
            else:
                obs.append(np.zeros(world.dim_p))
            obs.append(self.resource_encoding(r, world)) # len 4

        for c in self.consumers(world):
            if c.alive:
                obs.append(c.state.p_pos - agent.state.p_pos) # len 2
            else:
                obs.append(np.zeros(world.dim_p))
            obs.append(self.consumer_encoding(c, world)) # len 4

        return np.concatenate(obs)
        #24,24( 25:26, 31:32, 37:38, 43:44),12 (49:50, 55:56, )24,24,12, 
    def done(self,agent,world):
        return False
