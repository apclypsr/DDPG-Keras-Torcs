from gym_torcs import TorcsEnv
import numpy as np
import random
from dqn import DeepQNetwork
from generate_training_data_lou import MyConfig
from europilot.train import generate_training_data

import pickle
import argparse
from keras.models import model_from_json, Model
from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
import tensorflow as tf
#from keras.engine.training import collect_trainable_weights
import json

from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from OU import OU
import timeit

OU = OU()       #Ornstein-Uhlenbeck Process

def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def playGame(train_indicator=1):    #1 means Train, 0 means simply Run
    BUFFER_SIZE = 100000
    BATCH_SIZE = 32
    GAMMA = 0.99
    TAU = 0.001     #Target Network HyperParameters
    LRA = 0.0001    #Learning rate for Actor
    LRC = 0.001     #Lerning rate for Critic

    action_dim = 3  #Steering/Acceleration/Brake

    state_dim = 512

    #of sensors input

    np.random.seed(61502)

    vision = True

    EXPLORE = 100000.
    episode_count = 600000
    max_steps = 1800
    reward = 0
    done = False
    step = 0
    epsilon = 1
    indicator = 0
    esar2 = []
    esar4 = []

    #Tensorflow GPU optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)

    #We insert the Deep Q Image Processing Module
    args = {'save_model_freq' : 10000,
            'target_model_update_freq' : 10000,
            'normalize_weights' : True,
            'learning_rate' : .00025,
            'model' : None}

    # print(args["save_model_freq"])

    C= DeepQNetwork(state_dim, sess, '/home/lou/DDPG-Keras-Torcs', args=args)
    # print(C)

    x, h_fc1 = C.buildNetwork('test', trainable=True, numActions=1)

    actor = ActorNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA)
    critic = CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC)
    buff = ReplayBuffer(BUFFER_SIZE)    #Create replay buffer

    # Generate a Torcs environment
    env = TorcsEnv(vision=vision, throttle=True,gear_change=False)

    #Now load the weight
    print("Now we load the weight")
    try:
        actor.model.load_weights("actormodelIMG.h5")
        critic.model.load_weights("criticmodelIMG.h5")
        actor.target_model.load_weights("actormodelIMG.h5")
        critic.target_model.load_weights("criticmodelIMG.h5")
        print("Weight load successfully")
    except:
        print("Cannot find the weight")

    print("TORCS Experiment Start.")
    for i in range(episode_count):

        print("Episode : " + str(i) + " Replay Buffer " + str(buff.count()))

        if np.mod(i, 500) == 0:
            ob = env.reset(relaunch=True)   #relaunch TORCS every 500 episode because of the memory leak error
        else:
            ob = env.reset()

        imgfinal = np.zeros((1, 480, 640, 12), dtype=np.int32)
        s_t = C.getFC7(imgfinal)

        # print('ST FIRST', s_t)
        # print('STSHAPE FIRST', np.shape(s_t))

        total_reward = 0.

        imglst = []
        speed = 0
        stepreset = 1

        for j in range(max_steps):
            loss = 0 
            epsilon -= 1.0 / EXPLORE
            a_t = np.zeros([1,action_dim])

            noise_t = np.zeros([1,action_dim])
            
            # a_t_original = actor.model.predict(s_t.reshape(1, s_t.shape[0]))

            a_t_original = actor.model.predict(C.getFC7(imgfinal))
            # print('ATORIGINAL', a_t_original)
            noise_t[0][0] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][0],  0.0 , 0.60, 0.30)
            noise_t[0][1] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][1],  0.5 , 1.00, 0.10)
            noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2], -0.1 , 1.00, 0.05)

            #The following code do the stochastic brake
            if random.random() <= 0.05:
               print("********Now we apply the brake***********")
               noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2],  0.2 , 1.00, 0.10)

            a_t[0][0] = a_t_original[0][0] + noise_t[0][0]
            a_t[0][1] = a_t_original[0][1] + noise_t[0][1]
            a_t[0][2] = a_t_original[0][2] + noise_t[0][2]

            ob, r_t, done, info = env.step(a_t[0])

            # print('GTD1SUM:', np.sum(generate_training_data(config=MyConfig)))

            # print('GTD SHAPE', np.shape(generate_training_data(config=MyConfig)))
            imglst.append(generate_training_data(config=MyConfig))

            if len(imglst) == 4:
                imgcopy = imglst[:]
                imgfinal = np.stack(imgcopy)
                #print("Original stacked matrix", imgfinal)

                imgfinal = np.reshape(imgfinal, (4, 480, 640, 3))
                #print("Reshaped stacked matrix", imgfinal)

                #Switch 3 and 0 if you want to switch RGB or Batch
                imgfinal = np.transpose(imgfinal, (1,2,3,0))
                #print("Transposed stacked matrix", imgfinal)

                imgfinal = np.reshape(imgfinal, (1, 480, 640, 12))
                #print("Shape of imgfinal", imgfinal.shape)

            s_t1 = C.getFC7(imgfinal)

            #print('STL', s_t1)
            #print('STLSHAPE', np.shape(s_t1))
            #print('IMGFINAL', imgfinal)

            buff.add(s_t, a_t[0], r_t, s_t1, done)      #Add replay buffer

            #Do the batch update
            batch = buff.getBatch(BATCH_SIZE)
            states = np.asarray([e[0] for e in batch])

            # print('STATESSHAPE1', states)
            # print('SUMARRAY', np.sum(states))

            actions = np.asarray([e[1] for e in batch])
            rewards = np.asarray([e[2] for e in batch])
            new_states = np.asarray([e[3] for e in batch])
            dones = np.asarray([e[4] for e in batch])
            y_t = np.asarray([e[1] for e in batch])

            # print('NEW STATES', new_states)

            # target_q_values = critic.target_model.predict([C.getFC7(imgfinal), actor.target_model.predict(C.getFC7(imgfinal))])

            # print('ACTOR TARGET MODEL PREDICT', C.getFC7(imgfinal))
            new_states = np.reshape(new_states, (-1, state_dim))

            target_q_values = critic.target_model.predict([new_states, actor.target_model.predict(new_states)])
            # print('TARGET Q VALUES', target_q_values)
            # print('NEW STATES', new_states)
            # print('ACTOR MODEL PREDICT NEW STATES', actor.target_model.predict(new_states))
            # print('REWARDS', rewards)

            for k in range(len(batch)):
                if dones[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + GAMMA*target_q_values[k]

            if (train_indicator):
                states = np.reshape(states, (-1, state_dim))
                # print('STATESSHAPE2', np.shape(states))
                # print('ACTIONSSHAPE', np.shape(actions))
                # print('YT', np.shape(y_t))

                loss += critic.model.train_on_batch([states,actions], y_t) 
                a_for_grad = actor.model.predict(states)
                grads = critic.gradients(states, a_for_grad)
                actor.train(states, grads)
                actor.target_train()
                critic.target_train()

            total_reward += r_t
            s_t = s_t1
            speed += ob.speedX*300
            speedavg = speed/stepreset
            #print("SPEED X", ob.speedX)



            print("Episode", i, "Step", step, "Action", a_t, "Reward", r_t, "Loss", loss, "Average Speed", speedavg)
            esar = (i, step, a_t, r_t, loss, speedavg)
            esar2.append(esar)
        
            step += 1
            stepreset += 1

            if len(imglst) >= 4:
                del imglst[0]

            # print("Length of imglist", len(imglst))
            # print("List itself", imgfinal)

            if done:
                break

        if np.mod(i, 50) == 0:
            if (train_indicator):
                print("Now we save model")
                actor.model.save_weights("actormodelIMG.h5", overwrite=True)
                with open("actormodel.json", "w") as outfile:
                    json.dump(actor.model.to_json(), outfile)

                critic.model.save_weights("criticmodelIMG.h5", overwrite=True)
                with open("criticmodel.json", "w") as outfile:
                    json.dump(critic.model.to_json(), outfile)

        print("TOTAL REWARD @ " + str(i) +"-th Episode  : Reward " + str(total_reward))
        print("Total Step: " + str(step))
        print("")

        esar3 = (i, step, total_reward, speedavg)
        esar4.append(esar3)


        if np.mod(i, 50) == 0:
            save_object(esar2, 'IntraEpisode.pkl')
            save_object(esar4, 'InterEpisode.pkl')

    env.end()  # This is for shutting down TORCS
    print("Finish.")
    print("Saving esars.")

if __name__ == "__main__":
    playGame()