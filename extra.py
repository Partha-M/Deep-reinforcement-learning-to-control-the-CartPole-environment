for index in range(env.observation_space.n + 1):
    if index <= env.offsets[1]:
        Q[index, 4] = 0
    elif index <= env.offsets[2]:
        Q[index, 5] = 0
    elif index <= env.offsets[3]:
        Q[index, 6] = 0
    else:
        Q[index, 7] = 0
for index in range(env.observation_space.n + 1):
    if index <= env.offsets[1] - 1:
        Q[index, 8] = 0
    elif index <= env.offsets[2] - 1:
        Q[index, 9] = 0
    elif index <= env.offsets[3] - 1:
        Q[index, 10] = 0
    elif index <= env.observation_space.n - 1:
        Q[index, 11] = 0
    else:
        Q[index, 8] = 0