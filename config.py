class Game:
    Seed = 1234567
    MaxGameStep = 200
    AgentAction = 4
    AgentNum = 1
    total_time = 100

class Map:
    # Height = 6
    # Width = 6
    # source_pos = [[0,3],[2,0],[5,2],[3,5]]
    # hole_pos = [[2,2],[2,3],[3,2],[3,3]]
    # city_dis = [0.25,0.25,0.25,0.25]
    # hole_city = [0,1,2,3]
    # source_pos = [[0, 3], [2, 0], [5, 2], [3, 5]]
    # hole_pos = [[2, 2], [2, 3], [3, 2], [3, 3]]
    # city_dis = [0.25, 0.25, 0.25, 0.25]
    # hole_city = [0, 1, 2, 3]
    Height = 2
    Width = 5
    source_pos = [[0, 0]]
    hole_pos = [[4, 1]]
    city_dis = [1]
    hole_city = [0]

class Generator:
    RolloutSampleN = 10
    ExploreRate = 0.05

class StrongMazeEnv:
    ScoreLevel = 0.8
    EvaluateFile = '/tmp/evaluate.txt'

class Training:

    BatchSize = 32
    EnvBufferSize = 10000
    AgentBufferSize = 10000

    EnvEpsGen = 0.1
    RewardScaleGen = 1
    RewardScaleTrain = 1
    RewardScaleTest = 1

    EnvTrainEps = 1.0
    EnvTrainEps_Min = 0.2
    EnvTrainEps_HalfStep = 5000
    AgentTrainEps = 1.0
    AgentTrainEps_Min = 0.1
    AgentTrainEps_HalfStep = 2000

    EnvWarmup = 1000
    AgentWarmup = 1000

    EnvLearningRate = 1e-4
    AgentLearningRate = 1e-4

    EnvTargetModelUpdate = 1e-3
    AgentTargetModelUpdate = 1e-3


class Path:
    Figs = './figs'
    Logs = './logs'
    Models = './models'