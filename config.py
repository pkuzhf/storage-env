distances = None


class Game:
    Seed = 1234567
    AgentAction = 4
    AgentNum = 10
    total_time = 500

class Map:
    Height = 20
    Width = 20
    # source_pos = [[0,3],[2,0],[5,2],[3,5]]
    # hole_pos = [[2,2],[2,3],[3,2],[3,3]]
    # city_dis = [0.5, 0.5]
    # hole_city = [0, 1, 1, 0]
    # source_pos = [[0, 3], [2, 0], [5, 2], [3, 5]]
    # hole_pos = [[2, 2], [2, 3], [3, 2], [3, 3]]
    # city_dis = [0.25, 0.25, 0.25, 0.25]
    # hole_city = [0, 1, 2, 3]
    # Height = 3
    # Width = 5
    # source_pos = [[0, 0]]
    # hole_pos = [[4, 2]]
    # city_dis = [1]
    # hole_city = [0]
    city_dis = [0.26666666666666666, 0.13333333333333333, 0.2, 0.26666666666666666, 0.13333333333333333]
    source_pos = [[2, 2], [17, 2], [2, 5], [17, 5], [2, 8], [17, 8], [2, 11], [17, 11], [2, 14], [17, 14], [2, 17], [17, 17]]
    hole_pos = [[5, 3], [8, 3], [11, 3], [14, 3], [5, 6], [8, 6], [11, 6], [14, 6], [5, 9], [8, 9], [11, 9], [14, 9],
                [5, 12], [8, 12], [11, 12], [14, 12], [5, 15], [8, 15], [11, 15], [14, 15]]
    hole_city = [0, 2, 2, 4, 2, 3, 1, 4, 4, 3, 1, 2, 3, 0, 0, 2, 3, 3, 0, 3]


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