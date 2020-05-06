
# Importing the libraries
import numpy as np
from random import random, randint,randrange
import random
import datetime
import matplotlib.pyplot as plt
import time
import cv2
import os
import imutils
import torch
import torch.nn as nn
# Importing the Kivy packages
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.core.image import Image as CoreImage
from PIL import Image as PILImage
from kivy.graphics.texture import Texture
from kivy.properties import BoundedNumericProperty

# Importing the Dqn object from our AI in ai.py
from endgame_ai import TD3, ReplayBuffer

# Adding this line if we don't want the right click to put a red point
Config.set('graphics', 'multisamples', '0')
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
Config.set('graphics', 'resizable', False)
Config.set('graphics', 'width', '1429')
Config.set('graphics', 'height', '660')

# Introducing last_x and last_y, used to keep the last point in memory when we draw the sand on the map
last_x = 0
last_y = 0
n_points = 0
length = 0

#state_dim = (28,28,1)
state_dim = (1600)

action_dim = 1
max_action = 10

replay_buffer = ReplayBuffer()
policy = TD3(state_dim, action_dim, max_action)
action2rotation = [0,5,-5]
last_reward = 0
scores = []
im = CoreImage("./images/MASK1.png")
out_file = "./images/target.txt"
first_update = True

timestep = 0
max_steps = 1000
episode_timestep = 0
total_timestep = 10
episode_reward = 0 
episode_no = 0
eval_episodes = 10
done = True
episode_step=0
random_positions = {1:[137,385],2:[364,346],3:[582,310],4:[782,292],5:[1081,236],6:[338,179],7:[584,129],8:[671,442],9:[1104,351],10:[172,545],11:[245,101],12:[654,248],13:[804,181],14:[1146,157],15:[807,420],16:[712,440]}  #nk # nk17
car_prev_x = 597
car_prev_y = 71
seed = 0 # Random seed number
start_timesteps = 1e4 # Number of iterations/timesteps before which the model randomly chooses an action, and after which it starts to use the policy network
eval_freq = 5e3 # How often the evaluation step is performed (after how many timesteps)
max_timesteps = 5e5 # Total number of iterations/timesteps
save_models = True # Boolean checker whether or not to save the pre-trained model
expl_noise = 0.1 # Exploration noise - STD value of exploration Gaussian noise
batch_size = 100 # Size of the batch
discount = 0.99 # Discount factor gamma, used in the calculation of the total discounted reward
tau = 0.005 # Target network update rate
policy_noise = 0.2 # STD of Gaussian noise added to the actions for the exploration purposes
noise_clip = 0.5 # Maximum value of the Gaussian noise added to the actions (policy)
policy_freq = 2 # Number of iterations to wait before the policy network (Actor model) is updated


def init():
    global sand
    global goal_x
    global goal_y
    global first_update
    sand = np.zeros((longueur,largeur))
    img = PILImage.open("./images/mask.png").convert('L')
    sand = np.asarray(img)/255
    # print('sand values',sand)
    # print('sand shape',sand.shape)
    goal_x = 1090
    goal_y = 283
    first_update = False
    global swap
    swap = 0
    global car_prev_x 
    global car_prev_y 

last_distance = 0

# Creating the car class

class Ball1(Widget):
    pass
class Ball2(Widget):
    pass
class Ball3(Widget):
    pass

class Car(Widget):
    
    #angle = NumericProperty(0)
    #rotation = NumericProperty(0)
    angle = BoundedNumericProperty(0.0, min=- 180.0, max=180.0,errorhandler=lambda x: 180.0 if x > 180.0 else 0.0)
    #angle = BoundedNumericProperty(0.0)
    rotation = BoundedNumericProperty(0.0, min=- 30, max=30.0,errorhandler=lambda x: 30.0 if x > 300.0 else 0.0)
    #rotation = BoundedNumericProperty(0.0)
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)
    
    

    def move(self, rotation):
        #print("moving by",rotation)             - nk 27th Apr
        #print()
        self.pos = Vector(*self.velocity) + self.pos
        self.rotation = rotation
        self.angle = self.angle + self.rotation
        #print('the updated angle is', self.angle )              - nk 27th Apr
        # print(self.pos)



class Game(Widget):

    car = ObjectProperty(None)
    f= open("detail.txt","w+")
    def serve_car(self):
        self.car.center = self.center
        self.car.velocity = Vector(2, 0)

    def evaluate_policy(self,policy, eval_episodes=10):
        avg_reward = 0.
        for _ in range(eval_episodes):
            obs = 0
            done = False
            while not done:
                print("===================================")
                action = select_action(np.array(obs))
                obs, reward, done, _ = policy.take_step(action)
                avg_reward += reward
            avg_reward /= eval_episodes
        print ("---------------------------------------")
        print ("Average Reward over the Evaluation Step: %f" % (avg_reward))
        print ("---------------------------------------")
        return avg_reward
    #def make_3c(self):
    def make_3c(self):

        '''
        Function to take 3 image shots around the car and create 3 channels
        Added the orientation
        '''
        #defaulting to sand

        c3_state = np.ones((3,40,40))

        front_pos = Vector(0, 0).rotate(self.car.angle) + self.car.pos
        right_pos  = Vector(0, 0).rotate((self.car.angle+30)%360) + self.car.pos
        left_pos  = Vector(0, 0).rotate((self.car.angle-30)%360) + self.car.pos
        #if car_at_edge(self.x,self.y,self.width,self.height) :
        #    state = list(np.ones((3,20,20)).ravel())
        #else: 
        front_shot = sand[int(front_pos.x)-30:int(front_pos.x)+30, int(front_pos.y)-30:int(front_pos.y)+30]
        #print('=================',front_shot.shape)
        width = int(front_shot.shape[1] *(2/3) )
        height = int(front_shot.shape[0] * (2/3) )
        dimension = (width, height)
        if front_shot.shape != (60,60):
            front_shot = np.ones((40,40))
        else:    
            front_shot = cv2.resize(front_shot,dimension, interpolation = cv2.INTER_AREA)
        #print('**************',front_shot.shape)
        
        right_shot = sand[int(right_pos.x)-30:int(right_pos.x)+30, int(right_pos.y)-30:int(right_pos.y)+30]
        width = int(right_shot.shape[1] *(2/3) )
        height = int(right_shot.shape[0] *(2/3) )
        dimension = (width, height)
        if right_shot.shape != (60,60):
            right_shot = np.ones((40,40))
        else:
            right_shot = cv2.resize(right_shot,dimension, interpolation = cv2.INTER_AREA)
        
        left_shot  = sand[int(left_pos.x)-30:int(left_pos.x)+30, int(left_pos.y)-30:int(left_pos.y)+30]
        width = int(left_shot.shape[1] *(2/3) )
        height = int(left_shot.shape[0] *(2/3) )
        dimension = (width, height)
        if left_shot.shape != (60,60):
            left_shot = np.ones((40,40))
        else:
            left_shot = cv2.resize(left_shot,dimension, interpolation = cv2.INTER_AREA)
        
        try:
            c3_state = np.dstack((front_shot,right_shot,left_shot))
        except ValueError:
            c3_state = np.ones((3,40,40))

        #Padding when getting closer to the edges
        if c3_state.shape != (3,40,40) :
            c3_state = np.ones((3,40,40))
    
#         if front_shot.shape != (40,40):
#             front_shot = np.ones((40,40))
#         if left_shot.shape != (40,40):
#             left_shot = np.ones((40,40))
#         if right_shot.shape != (40,40):
#             right_shot = np.ones((40,40))
            
        return c3_state,front_shot,right_shot,left_shot
        
    def reset(self):

        reset_position = random_positions[randint(1,16)]
        self.f.write(str(reset_position))
        #reset_position = [car_prev_x,car_prev_y]
        self.car.x = reset_position[0] #arbitary position
        self.car.y = reset_position[1]
        #print(' after reset the car position is', self.car.pos)
        #self.car.angle = 0.0
        rescaled,front_shot,right_shot,left_shot = self.make_3c()
        #print("Car position in reset {} and {}".format(self.car.x,self.car.y))
        #img = sand[int(self.car.x):int(self.car.x)+60, int(self.car.y)-30 : int(self.car.y)+30]     # nk-1st july
        # image = cv2.imread('C:/Users/nihar/Downloads/endgame_nihar/endgame/images/mask_car.png')
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # overlay = imutils.rotate(image,self.car.angle)
        # rows,cols = overlay.shape
        # #overlay=cv2.addWeighted(img[30:30+rows, 30:30+cols],0.5,overlay,0.5,0)
        # img[30:30+rows, 30:30+cols ] = overlay
        #width = int(img.shape[1] *0.75 )        # nk-1st july
        #height = int(img.shape[0] *0.75 )       # nk-1st july
        #dimension = (width, height)
        #rescaled = cv2.resize(img,dimension, interpolation = cv2.INTER_AREA)
        return rescaled,front_shot,right_shot,left_shot
        #return sand[int(self.car.x)-14:int(self.car.x)+14, int(self.car.y)-14 : int(self.car.y)+14]

    def get_state(self):
        global car_prev_x
        global car_prev_y
        global done
        #send_state = sand[int(self.car.x)-14:int(self.car.x)+14, int(self.car.y)-14 : int(self.car.y)+14]
        #if send_state.shape == (28, 28):
        if int(self.car.x) > 29 and int(self.car.x) < 1400 and int(self.car.y) > 29 and int(self.car.y) < 631:  
            #print('gone from here up')          - nk 27th Apr
            #img = sand[int(self.car.x):int(self.car.x)+60, int(self.car.y)-30 : int(self.car.y)+30]
            # image = cv2.imread('C:/Users/nihar/Downloads/endgame_nihar/endgame/images/mask_car.png')
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # overlay = imutils.rotate(image,self.car.angle)
            # rows,cols = overlay.shape
            # #overlay=cv2.addWeighted(img[30:30+rows, 30:30+cols],0.5,overlay,0.5,0)
            # img[30:30+rows, 30:30+cols ] = overlay
            #width = int(img.shape[1] *0.75 )
            #height = int(img.shape[0] *0.75 )
            #dimension = (width, height)
            #rescaled = cv2.resize(img,dimension, interpolation = cv2.INTER_AREA)
            rescaled_state,front_shot,right_shot,left_shot = self.make_3c()
            return rescaled_state,front_shot,right_shot,left_shot
            #return sand[int(self.car.x)-14:int(self.car.x)+14, int(self.car.y)-14 : int(self.car.y)+14]
        else:
            #reset_position = random_positions[randint(1,16)]
            #self.car.x = reset_position[0]
            #self.car.y = reset_position[1]
            #print('gone from here up else ===================' ,reset_position)         - nk 27th Apr
            #img = sand[int(self.car.x):int(self.car.x)+60, int(self.car.y)-30 : int(self.car.y)+30]
            # image = cv2.imread('C:/Users/nihar/Downloads/endgame_nihar/endgame/images/mask_car.png')
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # overlay = imutils.rotate(image,self.car.angle)
            # rows,cols = overlay.shape
            # #overlay=cv2.addWeighted(img[30:30+rows, 30:30+cols],0.5,overlay,0.5,0)
            # img[30:30+rows, 30:30+cols ] = overlay
            #width = int(img.shape[1] *0.75 )
            #height = int(img.shape[0] *0.75 )
            #dimension = (width, height)
            #rescaled = cv2.resize(img,dimension, interpolation = cv2.INTER_AREA)
            #done = True 
            rescaled_state,front_shot,right_shot,left_shot = self.reset()
            return rescaled_state,front_shot,right_shot,left_shot
            #return sand[int(self.car.x)-14:int(self.car.x)+14,int(self.car.y)-14:int(self.car.y)+14]
        #else:
         #   return sand[int(self.car.x)-28:int(self.car.x)-14, int(self.car.y)-28 : int(self.car.y)-14]

    def take_step(self,action,last_distance,episode_step):
        global car_prev_x
        global car_prev_y
        global goal_x
        global goal_y
        global swap
        global done
        reward = 0
        once = 0
        
        rotation = action
        #print("=="*100)             - nk 27th Apr
        #print('last distance is', last_distance)
        self.car.move(rotation)
        img = sand[int(self.car.x)-30:int(self.car.x)+30, int(self.car.y)-30 : int(self.car.y)+30]
        # image = cv2.imread('C:/Users/nihar/Downloads/endgame_nihar/endgame/images/mask_car.png')
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # overlay = imutils.rotate(image,self.car.angle)
        # rows,cols = overlay.shape
        # #overlay=cv2.addWeighted(img[30:30+rows, 30:30+cols],0.5,overlay,0.5,0)
        # img[30:30+rows, 30:30+cols ] = overlay
        if int(self.car.x) > 29 and int(self.car.x) < 1400 and int(self.car.y) > 29 and int(self.car.y) < 631:
            #width = int(img.shape[1] *0.75)
            #height = int(img.shape[0] *0.75 )
            #dimension = (width, height)
            #rescaled = cv2.resize(img,dimension, interpolation = cv2.INTER_AREA)
            #new_state = rescaled
            new_state,front_shot,right_shot,left_shot = self.make_3c()
        else:
            new_state = img
            print("stuck near border so reseting to sand {}".format(new_state))
            reward += -10.0
            new_state,front_shot,right_shot,left_shot = self.make_3c()
        #new_state = sand[int(self.car.x)-14:int(self.car.x)+14, int(self.car.y)-14 : int(self.car.y)+14]
        '''
        if int(self.car.x) > 15 and int(self.car.x) < 1429 and int(self.car.y) > 15 and int(self.car.y) < 661:
            
            new_state = sand[int(self.car.x)-14:int(self.car.x)+14, int(self.car.y)-14 : int(self.car.y)+14]
        else:
            value =   random_positions[randint(2,15)]
            car_prev_x = value[0]
            car_prev_y = value[1]
            new_state = sand[int(car_prev_x)-14:int(car_prev_x)+14,int(car_prev_y)-14:int(car_prev_y)+14]
            #self.car.x = int(car_prev_x)
            #self.car.y = int(car_prev_x)
            
            print("stuck near border{}".format(new_state))
        '''
        distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)
        print(1, goal_x, goal_y, distance, int(self.car.x),int(self.car.y), im.read_pixel(int(self.car.x),int(self.car.y)))
        print('distance from goal is ', distance)
        if distance < 25: 
            if swap == 0:
                goal_x = 259
                goal_y = 372
                swap = 1
            else:
                goal_x =1090
                goal_y = 283
                swap = 0
            if once == 0:
                file1 = open(out_file, "a")
                once = 1
            write_text = "Target achieved" + str(datetime.datetime.now())
            file1.write(write_text) 
            
        #print("Car position in take_step {} and {}".format(self.car.x,self.car.y))          - nk 27th Apr
        #print('car position is',self.car.pos)
        if int(self.car.x) > 0 and int(self.car.x) < 1429 and int(self.car.y) > 0 and int(self.car.y) < 660:
            if sand[int(self.car.x),int(self.car.y)] > 0:
                print('Car in sand')
            
            else:
                print("Car on Road")
                car_prev_x = int(self.car.x)
                car_prev_y = int(self.car.y)
                print('x value {}'.format(car_prev_x))
                print('y value {}'.format(car_prev_y))
            
        
            episode_step += 1
            #print('episode step================',episode_step)              - nk 27th Apr
            if int(sand[int(self.car.x),int(self.car.y)]) > 0: 
                reward += -2.0
                self.car.velocity = Vector(1,0).rotate(self.car.angle)
                #done = True  #nk
            else :
                reward += -0.2
                self.car.velocity = Vector(4,0).rotate(self.car.angle)
                #done = False
        #print('just above done condition')              - nk 27th Apr
        
        if episode_step == 1000 and distance > 50:  #nk #nk16th again
            # episode_step = 0
            print('greater than 1000 and not near goal so let it learn')                  
            done = True
        elif episode_step == 1200:
            print('given 200 more chances but still not reached so let it run')
            done = True
            
        #last_distance = 1000   # Testing code nrk
        print(' the current distance from goal is', distance , 'and last distance was', last_distance)
        
        if distance > last_distance:
            print('=========== moving away from goal')
            reward += -1.0
        else:
            print('============ moving towards goal')
            reward += 1.5
            
#         if int(last_distance) > int(distance):   
#             print('moving away from goal')
#             reward += 1.5

        if self.car.x < 30:
            reset_position = random_positions[randint(1,16)]
            self.car.x = reset_position[0]
            self.car.y = reset_position[1]
            reward += -3
            #print('coming x10')             - nk 27th Apr
            #self.reset()
            #done = True
        if self.car.x > self.width - 30:
            reset_position = random_positions[randint(1,16)]
            self.car.x = reset_position[0]
            self.car.y = reset_position[1]
            #self.car.x = 580
            #self.car.y = 310
            reward += -3
            #print('coming x-10')                - nk 27th Apr
            #done = True
        if self.car.y < 30:
            reset_position = random_positions[randint(1,16)]
            self.car.x = reset_position[0]
            self.car.y = reset_position[1]
            #self.car.x = 580
            #self.car.y = 310
            reward += -3
            #print('coming y10')             - nk 27th Apr
            #done = True
        if self.car.y > self.height - 30:
            reset_position = random_positions[randint(1,16)]
            self.car.x = reset_position[0]
            self.car.y = reset_position[1]
            #self.car.x = 580
            #self.car.y = 310
            reward += -3
            #print('coming y-10')                - nk 27th Apr
            #done = True
        last_distance = distance

        return new_state,reward,done,episode_step,front_shot,right_shot,left_shot
    def overlay_car(self,state_c,front_shot,right_shot,left_shot):
        img1 = cv2.imread('C:/Users/nihar.kanungo/Downloads/endgame_nihar_28/endgame_nihar/endgame/images/mask_car.png')
        #print(image)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        #img1= np.dstack((img1,img1,img1))
        overlay = imutils.rotate(img1,self.car.angle)
        rows,cols = overlay.shape
        #print(overlay.shape)
        #state_c = np.dsplit(state_c,3)
        #state_c_front = state_c[0]
        #state_c_left = state_c[1]
        #state_c_right = state_c[2]
        #overlay=cv2.addWeighted(state_c[0:0+rows, 20:20+cols],0.5,overlay,0.5,0)
        front_shot[15:15+rows, 15:15+cols ] = overlay
        right_shot[15:15+rows, 15:15+cols ] = overlay
        left_shot[15:15+rows, 15:15+cols ] = overlay
        state_c = np.dstack((front_shot,right_shot,left_shot))
        
        #overlay=cv2.addWeighted(current_state[0:0+rows, 20:20+cols],0.5,overlay,0.5,0)
        #current_state[0:0+rows, 20:20+cols ] = overlay
        return state_c

    def update(self,dt):

        global policy
        global last_reward
        global scores
        global last_distance
        global goal_x
        global goal_y
        global longueur
        global largeur
        global swap
        global episode_timestep
        global total_timestep
        global replay_buffer
        global episode_reward 
        global done
        global discount
        global tau
        global policy_noise
        global noise_clip
        global policy_freq
        global timestep
        global episode_step
        global episode_no
        global eval_episodes
       #global self.angle
        #temp_state = np.ones((1,4800))
        longueur = self.width
        largeur = self.height

        if first_update:
            init()
        #xx = goal_x - self.car.x
        #yy = goal_y - self.car.y
        #orientation = Vector(*self.car.velocity).angle((xx,yy))/180.
        current_state,front_shot,right_shot,left_shot = self.get_state()
        #last_distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)
        file_name = "%s_%s_%s" % ("TD3", 'self_drive', episode_no +1)
        print ("---------------------------------------")
        print ("Settings: %s" % (file_name))
        print ("---------------------------------------")
        #print('timestep',timestep)          - nk 27th Apr
        
        if done:
            
            episode_no += 1
            
            #b = datetime.datetime.now()
            
            print('-------------------EPISODE DONE-------------------')
            if timestep!=0:
                a = datetime.datetime.now()
                print(' The time started training is',datetime.datetime.now())
                print('total episode:{},episode_timestep:{},episode reward:{},timestep:{}'.format(episode_no,episode_timestep,episode_reward,timestep))
                #image = cv2.imread('C:/Users/nihar/Downloads/endgame_nihar/endgame/images/mask_car.png')
                #print(image)
                policy.train(replay_buffer,episode_timestep,batch_size, discount, tau, policy_noise, noise_clip, policy_freq)
                b = datetime.datetime.now()
                print(' The Time taken to train',timestep, 'timesteps is =',b-a)
            # EVALUATION CONDITION AND CODE
            # print('angle here is', self.car.angle)             - nk 27th Apr
            current_state,front_shot,right_shot,left_shot = self.reset()
            done = False
            episode_reward = 0
            episode_timestep = 0
            episode_step = 0
        file_name = 'TD3' + '_' + str(episode_no)    
        if episode_no % 10 == 0:
        #if episode_no >= 1:        # For Testing removed - nk 27th Apr
           policy.save(file_name, directory="./pytorch_models")
        #     avg_reward = self.evaluate_policy(policy)
        #     np.save("./results/%s" % (file_name))
        xx = goal_x - self.car.x
        yy = goal_y - self.car.y
        orientation = Vector(*self.car.velocity).angle((xx,yy))/180.
        curr_state_orientation = [orientation, -orientation]
        last_distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)
        #current_state = self.get_state()
        current_state = self.overlay_car(current_state,front_shot,right_shot,left_shot)
        current_state = list(current_state.ravel())
        current_state .append(curr_state_orientation[0])
        current_state .append(curr_state_orientation[1])
        if timestep < 10000 : 
            #current_state = self.get_state()
            #action = max_action*randrange(-1,1)
            action = max_action*random.uniform(-1, 1)
        else:
            
          #  print('current state before calling action is', current_state.shape)
            #image = cv2.imread('C:/Users/nihar/Downloads/endgame_nihar/endgame/images/mask_car.png')
            action = policy.select_action(np.array(current_state))
        
        new_state,reward,done,episode_step ,front_shot,right_shot,left_shot= self.take_step(action,last_distance,episode_step)
        xx = goal_x - self.car.x
        yy = goal_y - self.car.y
        orientation = Vector(*self.car.velocity).angle((xx,yy))/180.
        next_state_orientation = [orientation, -orientation]
        #if episode_step == 500:
        #    episode_step = 0

        episode_reward += reward
        #print('the shape of state is', current_state.shape)
        #print('the shape of new state is', new_state.shape)   - nk 27th Apr
        #current_state = resize(current_state, (28, 28))
        #current_state = current_state.ravel()
        #overlay=cv2.addWeighted(new_state[0:0+rows, 20:20+cols],0.5,overlay,0.5,0)
        #new_state[0:0+rows, 20:20+cols ] = overlay
        if len(new_state.ravel()) == 4800:
            new_state = self.overlay_car(new_state,front_shot,right_shot,left_shot)
        new_state = new_state.ravel()
        #print('the shape of state is', current_state.shape)
        #print('the shape of new state is', new_state.shape)        - nk 27th Apr
        #print('the shape of state is', new_state.shape[0])         - nk 27th Apr
        #print('type of current state is', type(current_state))     - nk 27th Apr
        #print('type of temp state is', type(temp_state))           - nk 27th Apr
        if len(current_state) != 4800:
            #shape1 = new_state.shape[0]
            #temp_state[0][0:shape1] = new_state[0][0:shape1]
            current_state = np.ones((3,40,40)).ravel()
            #np.ones(image_size).ravel()
            #temp_state = np.ones((1, 1600))
        #print('the shape of state after is', new_state.shape)  - nk 27th Apr

        if len(new_state) != 4800:
            #shape1 = new_state.shape[0]
            #no_of_pixels = 1600 - new_state.shape[0]
            #temp_state[0][0:shape1] = new_state[0][0:shape1]
            new_state = np.ones((3,40,40)).ravel()
        #print('the shape of new state after is', new_state.shape)     - nk 27th Apr
        # img = PILImage.open("./images/mask_car.png")
        # img = img.rotate(self.car.angle)
        # #background = current_state
        # size = (40,40)
        # current_state = current_state.resize(size,PILImage.ANTIALIAS)
        # current_state.paste(img, (0,20), img)
        # new_state = new_state.resize(size,PILImage.ANTIALIAS)
        # new_state.paste(img, (0,20), img)
        # print(' i am able to properly superimpose')

        #current_state[0:0+rows, 20:20+cols ] = overlay
        current_state=list(current_state)
        current_state .append(curr_state_orientation[0])
        current_state .append(curr_state_orientation[1])
        #new_state[0:0+rows, 20:20+cols ] = overlay
        new_state = list(new_state)
        new_state .append(next_state_orientation[0])
        new_state .append(next_state_orientation[1])
        #new_state[785] = next_state_orientation[1]

        #print('the shape of action is', len(int(action)))
        #print('the shape of reward is', len(int(reward)))
        #print('the shape of done is', len(done))
        #current_state = np.reshape(current_state, (-1, 28))
        #print('the new shape of state is', len(current_state))          - nk 27th Apr
        #print('the new shape of state is', len(new_state))              - nk 27th Apr
        replay_buffer.add((current_state,new_state,action,reward,done))

        current_state = new_state
        episode_timestep += 1
        timestep += 1

        # Making a save method to save a trained model
    




        


class CarApp(App):

    def build(self):
        parent = Game()
        parent.serve_car()
        Clock.schedule_interval(parent.update, 1.0/60.0)
        # self.painter = MyPaintWidget()
        clearbtn = Button(text = 'clear')
        savebtn = Button(text = 'save', pos = (parent.width, 0))
        loadbtn = Button(text = 'load', pos = (2 * parent.width, 0))
        clearbtn.bind(on_release = self.clear_canvas)
        savebtn.bind(on_release = self.save)
        loadbtn.bind(on_release = self.load)
        # parent.add_widget(self.painter)
        parent.add_widget(clearbtn)
        parent.add_widget(savebtn)
        parent.add_widget(loadbtn)
        # parent.update(1.0/60.0)
        return parent

    def clear_canvas(self, obj):
        global sand

        # # self.painter.canvas.clear()
        # sand = np.zeros((longueur,largeur))

        

    def save(self, obj):
        print("saving brain...")
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))
        #plt.plot(scores)
        #plt.show()

    def load(self, obj):
        print("loading last saved brain...")
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))

# Running the whole thing
if __name__ == '__main__':
    CarApp().run()

