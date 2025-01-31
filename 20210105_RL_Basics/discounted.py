gamma = 0.99
#         s0  s1  s2
#       t: 0   1   2
rewards = [1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
# 1 * gamma^0 + 0 * gamma^1 + 1 * gamma^2
# 1 + 0 + 0.25 = 1.25


def discounted_reward(rewards, gamma):
    val = 0.0
    episode_length = len(rewards)
    print("Length: ", episode_length)
    #print("Test: ", 3**(2))

    for t in range(episode_length):
        val += gamma**(t) * rewards[t]
        print("Intermediate val: ", gamma**(t) * rewards[t])
        print("Val: ", val)

    return val


discounted_reward_value = discounted_reward(rewards, gamma)
print("Rewards: ", rewards)
print("Discounted Reward: ", discounted_reward_value)
