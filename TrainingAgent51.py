import tensorflow as tf
import numpy as np
import keras
from GymAgent import Agent
import os
import gym
#import TrainingAgent

tf.reset_default_graph()

#changed to BipedalWalker-v2 / CartPole-v1
env = gym.make("CartPole-v1")

discount_rate = 0.95

def discount_normalize_rewards(rewards):
    discounted_rewards = np.zeros_like(rewards)
    total_rewards = 0

    for i in reversed(range(len(rewards))):
        total_rewards = total_rewards * discount_rate + rewards[i]
        discounted_rewards[i] = total_rewards

    discounted_rewards -= np.mean(discounted_rewards)
    discounted_rewards /= np.std(discounted_rewards)

    return discounted_rewards
# Modify these to match shape of actions and states in your environment
#bipedal walker = 4 actions, and 24 statesize
num_actions = 2
state_size = 4

#change path to bipedal walker
path = "./cartpole-pg/"

training_episodes = 1000
max_steps_per_episode = 1000
episode_batch_size = 5

agent = Agent(num_actions, state_size)

init = tf.global_variables_initializer()

saver = tf.train.Saver(max_to_keep=2)

if not os.path.exists(path):
    os.makedirs(path)

with tf.Session() as sess:
    sess.run(init)

    total_episode_rewards = []

    # Create a buffer of 0'd gradients
    gradient_buffer = sess.run(tf.trainable_variables())
    for index, gradient in enumerate(gradient_buffer):
        gradient_buffer[index] = gradient * 0

    for episode in range(training_episodes):

        state = env.reset()

        episode_history = []
        episode_rewards = 0

        for step in range(max_steps_per_episode):

            if episode % 100 == 0:
                env.render()

            # Get weights for each action
            action_probabilities = sess.run(agent.outputs, feed_dict={agent.input_layer: [state]})
            action_choice = np.random.choice(range(num_actions), p=action_probabilities[0])

            state_next, reward, done, _ = env.step(action_choice)
            episode_history.append([state, action_choice, reward, state_next])
            state = state_next

            episode_rewards += reward

            if done or step + 1 == max_steps_per_episode:
                total_episode_rewards.append(episode_rewards)
                episode_history = np.array(episode_history)
                episode_history[:, 2] = discount_normalize_rewards(episode_history[:, 2])

                ep_gradients = sess.run(agent.gradients, feed_dict={agent.input_layer: np.vstack(episode_history[:, 0]),
                                                                    agent.actions: episode_history[:, 1],
                                                                    agent.rewards: episode_history[:, 2]})
                # add the gradients to the grad buffer:
                for index, gradient in enumerate(ep_gradients):
                    gradient_buffer[index] += gradient

                break

        #after the step for loop
        if episode % episode_batch_size == 0:
            feed_dict_gradients = dict(zip(agent.gradients_to_apply, gradient_buffer))
            sess.run(agent.update_gradients, feed_dict=feed_dict_gradients)
            for index, gradient in enumerate(gradient_buffer):
                gradient_buffer[index] = gradient * 0

            if episode % 100 == 0:
                saver.save(sess, path + "pg-checkpoint", episode)
                print("Average reward / 100 eps: " + str(np.mean(total_episode_rewards[-100:])))