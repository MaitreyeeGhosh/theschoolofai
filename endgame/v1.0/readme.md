
Assignment Requirement
==========================
Design and develop a self-driven car network which can train the car to move from one place to another with the following conditions
1.	The network must be a TD3 Network (Twin delayed DDPG)
2.	The map should be the one given and can’t be customized
3.	The Car is the given one, however the shape can be adjusted
4.	Can’t use any available environment, the map should behave as the environment
5.	The Car can’t have any sensors to sense the map and learn
6.	Need to use Convolutional Network for the model to train
7.	Should use KIVY as the GUI
8.	The car must have multiple destinations to reach
9.	The car should be trained to take the smallest number of steps to reach destination once trained
10.	Need to save the best video and share along with the code

List of Experiments performed 
================================

1.	Defined 16 different random locations for the agent to choose from incase of 
a.	Trying to move out of the map
b.	Not able to reach destination after taking 2000 steps
c.	If it reaches very near to the corners of the map
2.	Defined the action to be between -10 and +10 
3.	Take the state parameter in 3D 
        a.	One straight image of size 40 x 40 (scaled down from 60 x 60)
        b.	One left image after rotating the agent by an angle of 30 degrees of size 40 x 40 (scaled down from 60 x 60)
        c.	One right image after rotating the agent by an angle of 30 degrees of size 40 x 40 (scaled down from 60 x 60)
        Stacking three images together and send it as State parameter 
4.	Overlay a triangle shape for the car into the cropped image after rotating the car with an angle of self.car.angle 
5.	If the car goes to the edges of the map, then set Done = True and let the agent learn and assign -10 reward so that it          eventually learns not to go near the edges
6.	If the Distance to target less than the previous then give a reward to agent, but if it’s far then penalize 
7.	If the Agent on road then give a living penalty but if it’s on sand, then give a high penalty so that it eventually learns not to go on sand 
8.	If the Agent is < 25 pixels near t the target, then swap the target to the next one 
9.	If the agent on Road let it go faster and if on sand, go slower 
10.	After every 100 episodes remember to save the model 
11.	The First 10000 steps the agent should take random actions and after that it should take action from the TD3 Model
12.	Add +orientation and -orientation to the state parameter as well
13.	Save the parameters as a list to the buffer and resize it after reading from it 
14.	The Actor and Critic model consists of 
     a.	Several conv2d layers 
    b.	Batch Normalization 
    c.	Strides 
    
What I observed 
===================

# These observations are based on multiple runs of 50k steps. I could not run a full training step over 200-300k.

•	More number of channels in the convolution seems to give a better result but takes a lot of time to train.

•	If we set done as true every time it goes out of road the network sees very less example to train. However, over a period if it learns not to go out of road then it may mature itself to follow road only. I tried this one initially but as the processor is slow could not continue with that a lot. However, without this the car tries to move all around 

•	A higher reward for moving towards the target forces the agent to go towards target but not necessarily on road 

•	A balance to be made for reward for going towards target, penalty for moving away from target and penalty for going into sand. This is the key as any unbalanced reward mechanism may train the agent to choose one over another in which case the agent may not take optimal action

•	By giving more random locations to start, the agent seems to learn better than always starting from the same location

•	A very high penalty to go near the edges of the map initially forces the agent to go over the sands, but as the agent became more matured it started picking up the road while on edges rather on sand. But a perfect move seems to require much more training which I could not do.

•	I tried by capping the action between different values (eg: -50 50 , -30 30 , -20, 20 and -10 10, -5 5). I could not find out a major difference between -5 ,5 and -10, 10. I believe more training may help to choose the best capping for action 

•	Taking 3 images (front, left and right) seems to do better job than taking just the front view. But the real outcome may be seen while training more. Can’t prove a point at this time practically.

•	I could not figure out yet the difference between cropping a 40 x 40 versus cropping 60 x60 and scaling down to 40 x 40. But theoretically the later should work better. 

•	As the roads were at the edge of the map, hence allowed the car to go to edges but the next state became sands assuming that’s what I padded to the image all 4 sides to allow car to go to the roads at the corner .

•	Adding Orientation to state parameter seems to work better a little, but could not figure out the real difference. But using it 


 Videos
========= 

After several attempts I created this video before the doubt clearance class : https://youtu.be/oQMv2TxvzD4 This model trains but due to frequent done parameters it was not training properly .
Then I tried to modify the code, update the replay buffer save structure, updated Actor and Critic models , Taken care boundary conditions and added orientation . Here is the video . https://youtu.be/XtvPaOYTq7k
Video after training 50 k steps https://youtu.be/sxsV2L9E7_8

•	I have added the pending experiments to do towards the end 


Environment                                             Agent
=============                                         ===============  

![](images/citymap.png)                                  ![](images/car.png)

   
Code Description
======================

There are 3 Major programs in this assignment
1.	The car. kv : Which is the Kivy configuration file
2.	Endgame.ai.py – This contains the TD3 network and all associated methods
3.	Endgame_map.py – This is the main Game program which controls the agent by getting feedback from the environment

Endgame_map.py
==================== 
At first it imports all the necessary python and kivy libraries/packages into the program
Sets the image size Define the State Dimension, Action Dimension and Max action variable Create objects for replay buffer (this stores the agent’s experience) Create object of the TD3 network (this is the main network which decides the agent’s action) Load the map to a variable

The Initialization Function
===================================
Initializes the variable to hold the pixel values of the map Initializes the goal location Initializes all other variables

The Car Class
=====================
Initializes the variable to store Angle – Decides the car angle rotation - The Angle that the car takes from one state to another velocity – The speed of the car random position - this is to initialize car when it goes out start timesteps - Number of iterations/timesteps before which the model randomly chooses an action, and after which it starts to use the policy network evaluation frequency - ow often the evaluation step is performed (after how many timesteps) maximum timestep - Total number of iterations/timesteps exploration noise - Exploration noise - STD value of exploration Gaussian noise batch size - Size of the batch discount factor - Discount factor gamma, used in the calculation of the total discounted reward tau - Target network update rate policy noise - STD of Gaussian noise added to the actions for the exploration purposes noise clipping - Maximum value of the Gaussian noise added to the actions (policy) and policy frequency - Number of iterations to wait before the policy network (Actor model) is updated

The Move functions 
===================
moves the car the specified number of pixels

Game Class
========================
Creates a car object The serve Method Defines center of the car Initializes the velocity of the car The Evaluate policy function finds the action taken by the agent and based on that it calculates the average reward over the Evaluation Step The Reset Function resets the car back to a random location

The make_3c Function
=======================
Crops 3 images one in the front , one 30 degree right and another 30 degree left . Stacks it and returns a 3 channel image as the state 

Reset Function
=======================
Resets the car to random location

The Get state function
========================
crops the image around the car a specified pixel in both x and y axis and returns to the calling function The Take step function takes the action decided by the network and move the car accordingly. It also crops the image as the next state and based on the new location returns if the car is in road or sand. This also assigns the rewards based on the location of the car. It imposes heavy penalty for being on sand and a smaller living penalty for being on road. This also gives rewards for each step towards the goal and puts negative reward for it to going away from destination. This function also defined the destination locations for the car and returns new state, reward, episode step and done value

The Take step Function
=============================
Takes the action given by the network and returns the next state, reward and done. This is the function which calculates the reward mechanism and returns to the update function. This is the key function as the agent learns mostly from the reward and corrects itself.

Overlay Car Method 
========================
This method overlays the car shape (triangle in the self.car.x and self.car.y ) to the state.

The Update Function
=========================
If the Done parameter is true then it trains the agent by taking data from replay buffer. Resets Done to False Until the first 2000 (can be set as per programmer) timesteps the agent takes random action to fill the replay buffer and thereafter asks network to take action based on the current state. Calls the take step function to get the next state, reward, done and episode steps. Then it again adds the experience to replay buffer

Class CarApp
===================
It builds the game, saves the network and load last saved network Endgame_ai.py

Endgame_ai.py
=====================
Imports all python packages The Replay Buffer class executes 3 methods. The Init Method initializes storage, max size and pointer variable The Add method appends transitions to the storage which is nothing but the replay buffer The sample method creates list for each of the parameters (state, action, next state, reward and done) and returns the number of experiences equals to the batch size to the training function

The Actor Class
======================
Defines the network, here we are using Convolution layers as we are giving image to the network to learn and the forward function which performs the convolution and activation function. It also concatenates the orientation of the car which provides the angle to the destination. At the same time the max action amplifies the action which passes through the tanh function

The Critic Class
======================
The critic class also has init function which defines the critic network. Here we defined 2 critic networks as we are using 2 critics here. The forward function executes these 2 critic networks We also define a function Q1 which is nothing but the network which is the one which feeds to critic target.

TD3 Class
================
This is the brain of the network which defines how the agent will be trained by taking input from the network. Let’s discuss this step by step.
The Initialization method defines
1.	Actor model
2.	The Actor Target model
3.	Loads the Actor target dictionary
4.	Defines optimizer
5.	Sets critic model (This actually initiates 2 critics at the same time as this is what we have mentioned)
6.	Sets 2 Critic Targets
7.	Dictionary to load critic target
8.	Critic optimizer
9.	Max action

The Select Action Method
==============================
takes the current state and returns the action returned by the actor model The Train function loops through the iteration and for each It gets the sample from replay buffer (Current state, Action, Next State, Reward and done) Gets the next action from actor target which is required to train critic Adds random noise to the action Then gives next state and next action to both the critic targets to get target1 Q value and target2 Q value Then takes the minimum of the Q values in order to be less optimistic Adds discount factor to it and adds the reward received Then it takes the current state, current action and gives it to the critic model to get the Q values from critic model1 and critic model 2. Now it calculates the critic loss by taking the MSE loss of critic model1 vs critic target1 also adding that to MSE loss of critic model2 vs critic target2 Then the optimizer backpropagates the loss and optimizer optimize
Now the actor loss is also calculated and the actor model optimizes it self Still once every two iterations, we update the weights of the Actor target by polyak averaging and once every two iterations, we update the weights of the Critic target by polyak averaging We need to remember here that Every 2 times the target critic runs then the actor target gets updated and ever 2 full cycle runs the actor model gets updated. That means it’s very slow due to getting updated almost once in 6 times

Now let’s understand what we should see when running this program How to run?
Just go to your prompt and say “python endgame_map.py”

• Once the program runs and it allocates all necessary resources you should see a map and a car running over it.

• for the first few moment the car will take random actions and

• there after the network should give actions to the car 

• every time the car goes into sand, we will give it a negative feedback so that the car learns not to go into the sand

• every time the car goes to the edges of the map, we will give it a heavy penalty 

• Every time the car runs on road, we will give it a living penalty so that it doesn’t stop there for ling and try to reach goal as fast as possible 

• Every time the car moves away from goal, we will give it a negative penalty 

• Every time the car moves towards the goal the car will get some reward for it to know that it should continue moving in that direction 

• Now once the car reaches its goal the goal changes and the car keep moving to the new goal


• The process continues and the car keeps learning 

• Finally, we can stop the training and see how agent moves now


Improvements to make
========================
1.	Train car on road only, sand only and then append both for better learning
2.	Create own environment file and train on gpu 
3. Train for more number of steps 

