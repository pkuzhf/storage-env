distances = None
Type_num = 0
Source_num = 0
Hole_num = 0
directions = None

class Game:
    Seed = 1234567
    AgentAction = 4
    AgentNum = 100
    total_time = 400

class Map:
    Height = 50
    Width = 50
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
    # city_dis = [0.36666666666666666, 0.13333333333333333, 0.2, 0.26666666666666666, 0.03333333333333333]
    # source_pos = [[2, 2], [17, 2], [2, 5], [17, 5], [2, 8], [17, 8], [2, 11], [17, 11], [2, 14], [17, 14], [2, 17], [17, 17]]
    # hole_pos = [[5, 3], [8, 3], [11, 3], [14, 3], [5, 6], [8, 6], [11, 6], [14, 6], [5, 9], [8, 9], [11, 9], [14, 9],
    #             [5, 12], [8, 12], [11, 12], [14, 12], [5, 15], [8, 15], [11, 15], [14, 15]]
    # hole_city = [0, 2, 2, 4, 2, 3, 1, 4, 4, 3, 1, 2, 3, 0, 0, 2, 3, 3, 0, 3]
    # source_pos = [[2, 2], [97, 2], [2, 5], [97, 5], [2, 8], [97, 8], [2, 11], [97, 11], [2, 14], [97, 14], [2, 17], [97, 17],
    #  [2, 20], [97, 20], [2, 23], [97, 23], [2, 26], [97, 26], [2, 29], [97, 29], [2, 32], [97, 32], [2, 35], [97, 35],
    #  [2, 38], [97, 38], [2, 41], [97, 41], [2, 44], [97, 44], [2, 47], [97, 47], [2, 50], [97, 50], [2, 53], [97, 53],
    #  [2, 56], [97, 56], [2, 59], [97, 59], [2, 62], [97, 62], [2, 65], [97, 65], [2, 68], [97, 68], [2, 71], [97, 71],
    #  [2, 74], [97, 74], [2, 77], [97, 77], [2, 80], [97, 80], [2, 83], [97, 83], [2, 86], [97, 86], [2, 89], [97, 89],
    #  [2, 92], [97, 92], [2, 95], [97, 95]]
    # hole_pos = [[6, 2], [9, 2], [12, 2], [15, 2], [18, 2], [21, 2], [24, 2], [27, 2], [30, 2], [33, 2], [36, 2], [39, 2], [42, 2],
    #  [45, 2], [48, 2], [51, 2], [54, 2], [57, 2], [60, 2], [63, 2], [66, 2], [69, 2], [72, 2], [75, 2], [78, 2],
    #  [81, 2], [84, 2], [87, 2], [90, 2], [93, 2], [6, 5], [9, 5], [12, 5], [15, 5], [18, 5], [21, 5], [24, 5], [27, 5],
    #  [30, 5], [33, 5], [36, 5], [39, 5], [42, 5], [45, 5], [48, 5], [51, 5], [54, 5], [57, 5], [60, 5], [63, 5],
    #  [66, 5], [69, 5], [72, 5], [75, 5], [78, 5], [81, 5], [84, 5], [87, 5], [90, 5], [93, 5], [6, 8], [9, 8], [12, 8],
    #  [15, 8], [18, 8], [21, 8], [24, 8], [27, 8], [30, 8], [33, 8], [36, 8], [39, 8], [42, 8], [45, 8], [48, 8],
    #  [51, 8], [54, 8], [57, 8], [60, 8], [63, 8], [66, 8], [69, 8], [72, 8], [75, 8], [78, 8], [81, 8], [84, 8],
    #  [87, 8], [90, 8], [93, 8], [6, 11], [9, 11], [12, 11], [15, 11], [18, 11], [21, 11], [24, 11], [27, 11], [30, 11],
    #  [33, 11], [36, 11], [39, 11], [42, 11], [45, 11], [48, 11], [51, 11], [54, 11], [57, 11], [60, 11], [63, 11],
    #  [66, 11], [69, 11], [72, 11], [75, 11], [78, 11], [81, 11], [84, 11], [87, 11], [90, 11], [93, 11], [6, 14],
    #  [9, 14], [12, 14], [15, 14], [18, 14], [21, 14], [24, 14], [27, 14], [30, 14], [33, 14], [36, 14], [39, 14],
    #  [42, 14], [45, 14], [48, 14], [51, 14], [54, 14], [57, 14], [60, 14], [63, 14], [66, 14], [69, 14], [72, 14],
    #  [75, 14], [78, 14], [81, 14], [84, 14], [87, 14], [90, 14], [93, 14], [6, 17], [9, 17], [12, 17], [15, 17],
    #  [18, 17], [21, 17], [24, 17], [27, 17], [30, 17], [33, 17], [36, 17], [39, 17], [42, 17], [45, 17], [48, 17],
    #  [51, 17], [54, 17], [57, 17], [60, 17], [63, 17], [66, 17], [69, 17], [72, 17], [75, 17], [78, 17], [81, 17],
    #  [84, 17], [87, 17], [90, 17], [93, 17], [6, 20], [9, 20], [12, 20], [15, 20], [18, 20], [21, 20], [24, 20],
    #  [27, 20], [30, 20], [33, 20], [36, 20], [39, 20], [42, 20], [45, 20], [48, 20], [51, 20], [54, 20], [57, 20],
    #  [60, 20], [63, 20], [66, 20], [69, 20], [72, 20], [75, 20], [78, 20], [81, 20], [84, 20], [87, 20], [90, 20],
    #  [93, 20], [6, 23], [9, 23], [12, 23], [15, 23], [18, 23], [21, 23], [24, 23], [27, 23], [30, 23], [33, 23],
    #  [36, 23], [39, 23], [42, 23], [45, 23], [48, 23], [51, 23], [54, 23], [57, 23], [60, 23], [63, 23], [66, 23],
    #  [69, 23], [72, 23], [75, 23], [78, 23], [81, 23], [84, 23], [87, 23], [90, 23], [93, 23], [6, 26], [9, 26],
    #  [12, 26], [15, 26], [18, 26], [21, 26], [24, 26], [27, 26], [30, 26], [33, 26], [36, 26], [39, 26], [42, 26],
    #  [45, 26], [48, 26], [51, 26], [54, 26], [57, 26], [60, 26], [63, 26], [66, 26], [69, 26], [72, 26], [75, 26],
    #  [78, 26], [81, 26], [84, 26], [87, 26], [90, 26], [93, 26], [6, 29], [9, 29], [12, 29], [15, 29], [18, 29],
    #  [21, 29], [24, 29], [27, 29], [30, 29], [33, 29], [36, 29], [39, 29], [42, 29], [45, 29], [48, 29], [51, 29],
    #  [54, 29], [57, 29], [60, 29], [63, 29], [66, 29], [69, 29], [72, 29], [75, 29], [78, 29], [81, 29], [84, 29],
    #  [87, 29], [90, 29], [93, 29], [6, 32], [9, 32], [12, 32], [15, 32], [18, 32], [21, 32], [24, 32], [27, 32],
    #  [30, 32], [33, 32], [36, 32], [39, 32], [42, 32], [45, 32], [48, 32], [51, 32], [54, 32], [57, 32], [60, 32],
    #  [63, 32], [66, 32], [69, 32], [72, 32], [75, 32], [78, 32], [81, 32], [84, 32], [87, 32], [90, 32], [93, 32],
    #  [6, 35], [9, 35], [12, 35], [15, 35], [18, 35], [21, 35], [24, 35], [27, 35], [30, 35], [33, 35], [36, 35],
    #  [39, 35], [42, 35], [45, 35], [48, 35], [51, 35], [54, 35], [57, 35], [60, 35], [63, 35], [66, 35], [69, 35],
    #  [72, 35], [75, 35], [78, 35], [81, 35], [84, 35], [87, 35], [90, 35], [93, 35], [6, 38], [9, 38], [12, 38],
    #  [15, 38], [18, 38], [21, 38], [24, 38], [27, 38], [30, 38], [33, 38], [36, 38], [39, 38], [42, 38], [45, 38],
    #  [48, 38], [51, 38], [54, 38], [57, 38], [60, 38], [63, 38], [66, 38], [69, 38], [72, 38], [75, 38], [78, 38],
    #  [81, 38], [84, 38], [87, 38], [90, 38], [93, 38], [6, 41], [9, 41], [12, 41], [15, 41], [18, 41], [21, 41],
    #  [24, 41], [27, 41], [30, 41], [33, 41], [36, 41], [39, 41], [42, 41], [45, 41], [48, 41], [51, 41], [54, 41],
    #  [57, 41], [60, 41], [63, 41], [66, 41], [69, 41], [72, 41], [75, 41], [78, 41], [81, 41], [84, 41], [87, 41],
    #  [90, 41], [93, 41], [6, 44], [9, 44], [12, 44], [15, 44], [18, 44], [21, 44], [24, 44], [27, 44], [30, 44],
    #  [33, 44], [36, 44], [39, 44], [42, 44], [45, 44], [48, 44], [51, 44], [54, 44], [57, 44], [60, 44], [63, 44],
    #  [66, 44], [69, 44], [72, 44], [75, 44], [78, 44], [81, 44], [84, 44], [87, 44], [90, 44], [93, 44], [6, 47],
    #  [9, 47], [12, 47], [15, 47], [18, 47], [21, 47], [24, 47], [27, 47], [30, 47], [33, 47], [36, 47], [39, 47],
    #  [42, 47], [45, 47], [48, 47], [51, 47], [54, 47], [57, 47], [60, 47], [63, 47], [66, 47], [69, 47], [72, 47],
    #  [75, 47], [78, 47], [81, 47], [84, 47], [87, 47], [90, 47], [93, 47], [6, 50], [9, 50], [12, 50], [15, 50],
    #  [18, 50], [21, 50], [24, 50], [27, 50], [30, 50], [33, 50], [36, 50], [39, 50], [42, 50], [45, 50], [48, 50],
    #  [51, 50], [54, 50], [57, 50], [60, 50], [63, 50], [66, 50], [69, 50], [72, 50], [75, 50], [78, 50], [81, 50],
    #  [84, 50], [87, 50], [90, 50], [93, 50], [6, 53], [9, 53], [12, 53], [15, 53], [18, 53], [21, 53], [24, 53],
    #  [27, 53], [30, 53], [33, 53], [36, 53], [39, 53], [42, 53], [45, 53], [48, 53], [51, 53], [54, 53], [57, 53],
    #  [60, 53], [63, 53], [66, 53], [69, 53], [72, 53], [75, 53], [78, 53], [81, 53], [84, 53], [87, 53], [90, 53],
    #  [93, 53], [6, 56], [9, 56], [12, 56], [15, 56], [18, 56], [21, 56], [24, 56], [27, 56], [30, 56], [33, 56],
    #  [36, 56], [39, 56], [42, 56], [45, 56], [48, 56], [51, 56], [54, 56], [57, 56], [60, 56], [63, 56], [66, 56],
    #  [69, 56], [72, 56], [75, 56], [78, 56], [81, 56], [84, 56], [87, 56], [90, 56], [93, 56], [6, 59], [9, 59],
    #  [12, 59], [15, 59], [18, 59], [21, 59], [24, 59], [27, 59], [30, 59], [33, 59], [36, 59], [39, 59], [42, 59],
    #  [45, 59], [48, 59], [51, 59], [54, 59], [57, 59], [60, 59], [63, 59], [66, 59], [69, 59], [72, 59], [75, 59],
    #  [78, 59], [81, 59], [84, 59], [87, 59], [90, 59], [93, 59], [6, 62], [9, 62], [12, 62], [15, 62], [18, 62],
    #  [21, 62], [24, 62], [27, 62], [30, 62], [33, 62], [36, 62], [39, 62], [42, 62], [45, 62], [48, 62], [51, 62],
    #  [54, 62], [57, 62], [60, 62], [63, 62], [66, 62], [69, 62], [72, 62], [75, 62], [78, 62], [81, 62], [84, 62],
    #  [87, 62], [90, 62], [93, 62], [6, 65], [9, 65], [12, 65], [15, 65], [18, 65], [21, 65], [24, 65], [27, 65],
    #  [30, 65], [33, 65], [36, 65], [39, 65], [42, 65], [45, 65], [48, 65], [51, 65], [54, 65], [57, 65], [60, 65],
    #  [63, 65], [66, 65], [69, 65], [72, 65], [75, 65], [78, 65], [81, 65], [84, 65], [87, 65], [90, 65], [93, 65],
    #  [6, 68], [9, 68], [12, 68], [15, 68], [18, 68], [21, 68], [24, 68], [27, 68], [30, 68], [33, 68], [36, 68],
    #  [39, 68], [42, 68], [45, 68], [48, 68], [51, 68], [54, 68], [57, 68], [60, 68], [63, 68], [66, 68], [69, 68],
    #  [72, 68], [75, 68], [78, 68], [81, 68], [84, 68], [87, 68], [90, 68], [93, 68], [6, 71], [9, 71], [12, 71],
    #  [15, 71], [18, 71], [21, 71], [24, 71], [27, 71], [30, 71], [33, 71], [36, 71], [39, 71], [42, 71], [45, 71],
    #  [48, 71], [51, 71], [54, 71], [57, 71], [60, 71], [63, 71], [66, 71], [69, 71], [72, 71], [75, 71], [78, 71],
    #  [81, 71], [84, 71], [87, 71], [90, 71], [93, 71], [6, 74], [9, 74], [12, 74], [15, 74], [18, 74], [21, 74],
    #  [24, 74], [27, 74], [30, 74], [33, 74], [36, 74], [39, 74], [42, 74], [45, 74], [48, 74], [51, 74], [54, 74],
    #  [57, 74], [60, 74], [63, 74], [66, 74], [69, 74], [72, 74], [75, 74], [78, 74], [81, 74], [84, 74], [87, 74],
    #  [90, 74], [93, 74], [6, 77], [9, 77], [12, 77], [15, 77], [18, 77], [21, 77], [24, 77], [27, 77], [30, 77],
    #  [33, 77], [36, 77], [39, 77], [42, 77], [45, 77], [48, 77], [51, 77], [54, 77], [57, 77], [60, 77], [63, 77],
    #  [66, 77], [69, 77], [72, 77], [75, 77], [78, 77], [81, 77], [84, 77], [87, 77], [90, 77], [93, 77], [6, 80],
    #  [9, 80], [12, 80], [15, 80], [18, 80], [21, 80], [24, 80], [27, 80], [30, 80], [33, 80], [36, 80], [39, 80],
    #  [42, 80], [45, 80], [48, 80], [51, 80], [54, 80], [57, 80], [60, 80], [63, 80], [66, 80], [69, 80], [72, 80],
    #  [75, 80], [78, 80], [81, 80], [84, 80], [87, 80], [90, 80], [93, 80], [6, 83], [9, 83], [12, 83], [15, 83],
    #  [18, 83], [21, 83], [24, 83], [27, 83], [30, 83], [33, 83], [36, 83], [39, 83], [42, 83], [45, 83], [48, 83],
    #  [51, 83], [54, 83], [57, 83], [60, 83], [63, 83], [66, 83], [69, 83], [72, 83], [75, 83], [78, 83], [81, 83],
    #  [84, 83], [87, 83], [90, 83], [93, 83], [6, 86], [9, 86], [12, 86], [15, 86], [18, 86], [21, 86], [24, 86],
    #  [27, 86], [30, 86], [33, 86], [36, 86], [39, 86], [42, 86], [45, 86], [48, 86], [51, 86], [54, 86], [57, 86],
    #  [60, 86], [63, 86], [66, 86], [69, 86], [72, 86], [75, 86], [78, 86], [81, 86], [84, 86], [87, 86], [90, 86],
    #  [93, 86], [6, 89], [9, 89], [12, 89], [15, 89], [18, 89], [21, 89], [24, 89], [27, 89], [30, 89], [33, 89],
    #  [36, 89], [39, 89], [42, 89], [45, 89], [48, 89], [51, 89], [54, 89], [57, 89], [60, 89], [63, 89], [66, 89],
    #  [69, 89], [72, 89], [75, 89], [78, 89], [81, 89], [84, 89], [87, 89], [90, 89], [93, 89], [6, 92], [9, 92],
    #  [12, 92], [15, 92], [18, 92], [21, 92], [24, 92], [27, 92], [30, 92], [33, 92], [36, 92], [39, 92], [42, 92],
    #  [45, 92], [48, 92], [51, 92], [54, 92], [57, 92], [60, 92], [63, 92], [66, 92], [69, 92], [72, 92], [75, 92],
    #  [78, 92], [81, 92], [84, 92], [87, 92], [90, 92], [93, 92], [6, 95], [9, 95], [12, 95], [15, 95], [18, 95],
    #  [21, 95], [24, 95], [27, 95], [30, 95], [33, 95], [36, 95], [39, 95], [42, 95], [45, 95], [48, 95], [51, 95],
    #  [54, 95], [57, 95], [60, 95], [63, 95], [66, 95], [69, 95], [72, 95], [75, 95], [78, 95], [81, 95], [84, 95],
    #  [87, 95], [90, 95], [93, 95]]
    # hole_city = [78, 91, 27, 54, 26, 15, 87, 56, 90, 75, 77, 22, 98, 36, 20, 95, 48, 0, 42, 93, 62, 76, 69, 19, 28, 32, 44, 40, 71,
    #  88, 30, 5, 72, 11, 99, 39, 18, 9, 81, 31, 12, 14, 38, 1, 83, 59, 46, 34, 33, 17, 2, 67, 29, 92, 45, 58, 86, 73, 61,
    #  50, 79, 64, 3, 96, 66, 63, 7, 16, 4, 51, 10, 8, 24, 82, 89, 68, 65, 43, 37, 47, 94, 21, 35, 60, 6, 25, 49, 97, 55,
    #  74, 52, 23, 70, 41, 57, 53, 84, 80, 85, 13, 0, 1, 7, 10, 43, 18, 54, 12, 4, 7, 21, 3, 32, 8, 1, 62, 8, 0, 4, 24, 1,
    #  6, 38, 72, 55, 9, 8, 36, 60, 78, 4, 10, 0, 21, 0, 37, 7, 28, 32, 29, 5, 1, 52, 53, 15, 1, 0, 10, 8, 39, 1, 10, 50,
    #  6, 2, 0, 1, 41, 7, 20, 0, 15, 6, 4, 16, 42, 14, 56, 67, 20, 0, 13, 0, 51, 12, 1, 54, 9, 2, 22, 2, 31, 52, 6, 5, 76,
    #  41, 0, 11, 14, 48, 7, 3, 50, 0, 0, 32, 22, 3, 10, 5, 56, 5, 9, 0, 6, 8, 29, 24, 5, 60, 7, 14, 0, 13, 0, 22, 0, 1,
    #  29, 13, 38, 25, 75, 44, 36, 1, 0, 1, 98, 74, 75, 1, 19, 3, 3, 42, 2, 7, 2, 6, 0, 57, 6, 41, 58, 13, 62, 3, 4, 11,
    #  12, 10, 1, 48, 11, 96, 33, 15, 15, 11, 77, 83, 8, 7, 4, 62, 1, 31, 17, 7, 5, 53, 3, 1, 4, 14, 3, 4, 10, 6, 1, 4,
    #  29, 2, 63, 0, 8, 63, 12, 3, 3, 8, 43, 9, 28, 45, 1, 39, 0, 88, 24, 9, 0, 9, 3, 79, 1, 6, 4, 19, 12, 34, 24, 27, 2,
    #  57, 42, 32, 89, 5, 3, 10, 7, 1, 79, 2, 74, 12, 13, 3, 3, 15, 85, 5, 80, 40, 1, 77, 4, 1, 19, 57, 19, 68, 1, 41, 35,
    #  68, 21, 8, 0, 52, 0, 5, 0, 35, 1, 9, 55, 7, 13, 49, 25, 7, 4, 27, 9, 49, 4, 2, 0, 2, 7, 36, 4, 37, 3, 24, 1, 5, 17,
    #  27, 4, 2, 2, 24, 21, 1, 28, 11, 0, 16, 19, 0, 12, 23, 1, 2, 95, 16, 2, 0, 25, 1, 0, 20, 1, 1, 21, 92, 2, 25, 93,
    #  94, 6, 17, 10, 4, 5, 3, 53, 38, 6, 9, 2, 0, 98, 3, 11, 33, 3, 23, 7, 4, 44, 25, 6, 13, 2, 2, 11, 3, 1, 17, 29, 6,
    #  0, 34, 26, 85, 51, 22, 65, 30, 6, 21, 12, 83, 27, 0, 95, 5, 2, 12, 10, 14, 21, 1, 25, 10, 0, 2, 10, 32, 23, 10, 0,
    #  3, 14, 73, 4, 6, 3, 40, 36, 24, 3, 33, 8, 34, 2, 2, 5, 50, 0, 15, 18, 8, 8, 6, 7, 27, 5, 34, 5, 27, 1, 16, 2, 36,
    #  3, 62, 0, 6, 7, 7, 6, 15, 4, 14, 3, 77, 1, 12, 5, 0, 8, 68, 1, 30, 45, 6, 60, 32, 1, 14, 83, 2, 66, 2, 33, 69, 56,
    #  0, 6, 89, 51, 0, 2, 20, 23, 28, 2, 4, 45, 23, 0, 0, 11, 36, 4, 1, 22, 24, 1, 7, 46, 27, 0, 3, 0, 19, 40, 94, 3, 2,
    #  42, 13, 2, 3, 97, 19, 0, 38, 11, 0, 66, 11, 0, 6, 78, 7, 29, 54, 7, 0, 0, 3, 52, 43, 0, 38, 82, 8, 0, 6, 26, 59, 0,
    #  15, 16, 18, 10, 30, 21, 67, 80, 0, 80, 77, 0, 40, 11, 22, 79, 1, 52, 0, 0, 1, 26, 4, 4, 3, 70, 26, 0, 54, 89, 14,
    #  12, 5, 1, 20, 47, 69, 10, 29, 13, 43, 2, 0, 25, 4, 26, 6, 39, 0, 1, 11, 4, 25, 9, 97, 11, 0, 22, 32, 66, 4, 57, 5,
    #  12, 0, 0, 3, 1, 26, 4, 1, 35, 62, 39, 0, 2, 10, 2, 98, 14, 30, 5, 8, 23, 20, 99, 81, 13, 19, 63, 4, 99, 9, 1, 4, 1,
    #  15, 69, 43, 5, 0, 0, 0, 1, 14, 21, 39, 1, 0, 4, 22, 15, 0, 3, 0, 0, 10, 48, 42, 76, 41, 10, 2, 0, 0, 23, 89, 3, 33,
    #  0, 87, 50, 11, 47, 44, 32, 1, 22, 50, 10, 2, 23, 54, 21, 46, 12, 5, 2, 37, 0, 40, 1, 18, 53, 76, 2, 1, 30, 31, 26,
    #  48, 68, 1, 22, 0, 3, 7, 0, 15, 52, 7, 3, 66, 9, 0, 83, 2, 17, 12, 46, 90, 42, 2, 6, 56, 7, 57, 26, 73, 34, 72, 1,
    #  0, 2, 70, 25, 44, 10, 29, 27, 4, 1, 18, 5, 86, 1, 1, 16, 7, 0, 2, 1, 58, 31, 36, 43, 0, 2, 34, 74, 22, 8, 0, 67, 5,
    #  6, 19, 1, 48, 1, 62, 35, 25, 77, 1, 0, 2, 61, 14, 7, 3, 9, 6, 14, 46, 0, 2, 14, 4, 0, 9, 1, 67, 12, 5, 2, 0, 0, 81,
    #  1, 51, 19, 0, 7, 74, 9, 21, 3, 6, 0, 3, 2, 38, 10, 2, 2, 46, 50, 29, 4, 90, 7, 25, 5, 35, 1, 0, 19, 52, 6, 1, 58,
    #  48, 2, 4, 15, 70, 6, 3, 34, 51, 43, 55, 29, 17, 14, 14, 51, 40, 19, 2, 84, 46, 91, 7, 2, 35, 0, 0, 33, 6, 9, 14, 9,
    #  46, 21, 6, 53, 17, 46, 3, 3, 35, 0, 3, 0, 39, 28, 0, 6, 1, 20, 55, 39, 6, 51, 0, 83, 0, 0]
    # city_dis = [0.10199345760440795, 0.07285246971743424, 0.05666303200244885, 0.04636066254745816, 0.039228252924772285,
    #  0.03399781920146931, 0.029998075766002334, 0.026840383580107353, 0.02428415657247808, 0.02217249078356694,
    #  0.020398691520881588, 0.01888767733414962, 0.01758507889731171, 0.016450557678130313, 0.015453554182486052,
    #  0.014570493943486851, 0.013782899676271346, 0.01307608430825743, 0.01243822653712292, 0.011859704372605577,
    #  0.011332606400489772, 0.010850367830256164, 0.010407495673919178, 0.009999358588667446, 0.009622024302302638,
    #  0.009272132509491632, 0.00894679452670245, 0.008643513356305758, 0.008360119475771144, 0.008094718857492696,
    #  0.007845650584954457, 0.0076114520600304445, 0.007390830261188982, 0.007182637859465348, 0.006985853260575887,
    #  0.006799563840293862, 0.006622951792494022, 0.006455282126861263, 0.0062958924447165405, 0.006144184193036624,
    #  0.005999615153200468, 0.005861692965770571, 0.005729969528337525, 0.005604036132110326, 0.005483519226043438,
    #  0.0053680767160214716, 0.0052573947218766985, 0.005151184727495351, 0.005049181069525146, 0.004951138718660581,
    #  0.004856831314495616, 0.004766049420766726, 0.004678598972679263, 0.004594299892090449, 0.004512984849752565,
    #  0.004434498156713389, 0.004358694769419143, 0.004285439395143191, 0.004214605686132559, 0.004146075512374307,
    #  0.004079738304176318, 0.004015490456866455, 0.003953234790868525, 0.0038928800612369444, 0.0038343405114439084,
    #  0.003777535466829924, 0.0037223889636645233, 0.0036688294102305017, 0.0036167892767520545, 0.003566204811342935,
    #  0.0035170157794623424, 0.003469165224639726, 0.0034225992484700645, 0.003377266808092978, 0.0033331195295558153,
    #  0.0032901115356260622, 0.003248199286764584, 0.0032073414341008783, 0.003167498683366706, 0.003128633668846869,
    #  0.0030907108364972105, 0.0030536963354613157, 0.0030175579172901754, 0.0029822648422341506, 0.002947787792034911,
    #  0.0029140987886973695, 0.0028811711187685858, 0.002848979262692959, 0.002817498828851048, 0.002786706491923714,
    #  0.002756579935254269, 0.0027270977969093032, 0.002698239619164231, 0.0026699858011625115, 0.0026423175545183405,
    #  0.0026152168616514855, 0.0025886664366601, 0.0025626496885529635, 0.0025371506866768145, 0.0025121541281873877]

    city_dis = [0.15427709917835827, 0.11019792798454162, 0.08570949954353237, 0.07012595417198104, 0.0593373458378301,
     0.05142569972611943, 0.04537561740539949, 0.04059923662588376, 0.036732642661513874, 0.033538499821382226,
     0.030855419835671652, 0.02856983318117746, 0.026599499858337632, 0.02488340309328359, 0.023375318057327014,
     0.02203958559690833, 0.020848256645724095, 0.019779115279276702, 0.018814280387604666, 0.017939197578878872,
     0.017141899908706477, 0.016412457359399817, 0.015742561140648804, 0.015125205801799834, 0.014554443318713046]
    source_pos = [[2, 2], [47, 2], [2, 5], [47, 5], [2, 8], [47, 8], [2, 11], [47, 11], [2, 14], [47, 14], [2, 17], [47, 17],
     [2, 20], [47, 20], [2, 23], [47, 23], [2, 26], [47, 26], [2, 29], [47, 29], [2, 32], [47, 32], [2, 35], [47, 35],
     [2, 38], [47, 38], [2, 41], [47, 41], [2, 44], [47, 44], [2, 47], [47, 47]]
    hole_pos = [[5, 3], [8, 3], [11, 3], [14, 3], [17, 3], [20, 3], [23, 3], [26, 3], [29, 3], [32, 3], [35, 3], [38, 3], [41, 3],
     [44, 3], [5, 6], [8, 6], [11, 6], [14, 6], [17, 6], [20, 6], [23, 6], [26, 6], [29, 6], [32, 6], [35, 6], [38, 6],
     [41, 6], [44, 6], [5, 9], [8, 9], [11, 9], [14, 9], [17, 9], [20, 9], [23, 9], [26, 9], [29, 9], [32, 9], [35, 9],
     [38, 9], [41, 9], [44, 9], [5, 12], [8, 12], [11, 12], [14, 12], [17, 12], [20, 12], [23, 12], [26, 12], [29, 12],
     [32, 12], [35, 12], [38, 12], [41, 12], [44, 12], [5, 15], [8, 15], [11, 15], [14, 15], [17, 15], [20, 15],
     [23, 15], [26, 15], [29, 15], [32, 15], [35, 15], [38, 15], [41, 15], [44, 15], [5, 18], [8, 18], [11, 18],
     [14, 18], [17, 18], [20, 18], [23, 18], [26, 18], [29, 18], [32, 18], [35, 18], [38, 18], [41, 18], [44, 18],
     [5, 21], [8, 21], [11, 21], [14, 21], [17, 21], [20, 21], [23, 21], [26, 21], [29, 21], [32, 21], [35, 21],
     [38, 21], [41, 21], [44, 21], [5, 24], [8, 24], [11, 24], [14, 24], [17, 24], [20, 24], [23, 24], [26, 24],
     [29, 24], [32, 24], [35, 24], [38, 24], [41, 24], [44, 24], [5, 27], [8, 27], [11, 27], [14, 27], [17, 27],
     [20, 27], [23, 27], [26, 27], [29, 27], [32, 27], [35, 27], [38, 27], [41, 27], [44, 27], [5, 30], [8, 30],
     [11, 30], [14, 30], [17, 30], [20, 30], [23, 30], [26, 30], [29, 30], [32, 30], [35, 30], [38, 30], [41, 30],
     [44, 30], [5, 33], [8, 33], [11, 33], [14, 33], [17, 33], [20, 33], [23, 33], [26, 33], [29, 33], [32, 33],
     [35, 33], [38, 33], [41, 33], [44, 33], [5, 36], [8, 36], [11, 36], [14, 36], [17, 36], [20, 36], [23, 36],
     [26, 36], [29, 36], [32, 36], [35, 36], [38, 36], [41, 36], [44, 36], [5, 39], [8, 39], [11, 39], [14, 39],
     [17, 39], [20, 39], [23, 39], [26, 39], [29, 39], [32, 39], [35, 39], [38, 39], [41, 39], [44, 39], [5, 42],
     [8, 42], [11, 42], [14, 42], [17, 42], [20, 42], [23, 42], [26, 42], [29, 42], [32, 42], [35, 42], [38, 42],
     [41, 42], [44, 42], [5, 45], [8, 45], [11, 45], [14, 45], [17, 45], [20, 45], [23, 45], [26, 45], [29, 45],
     [32, 45], [35, 45], [38, 45], [41, 45], [44, 45]]
    hole_city = [13, 12, 17, 3, 16, 18, 15, 22, 5, 14, 21, 10, 0, 2, 23, 9, 19, 11, 4, 7, 1, 20, 6, 8, 24, 1, 3, 16, 3, 14, 17, 9,
     0, 6, 0, 1, 4, 1, 0, 0, 5, 16, 4, 6, 10, 0, 12, 0, 12, 0, 0, 19, 0, 5, 16, 12, 0, 0, 1, 4, 3, 14, 1, 2, 4, 4, 1, 9,
     2, 3, 2, 6, 7, 2, 4, 11, 0, 0, 18, 4, 1, 4, 2, 0, 1, 21, 1, 13, 0, 4, 7, 22, 1, 8, 2, 1, 4, 12, 0, 0, 2, 1, 0, 9,
     11, 1, 2, 21, 8, 10, 1, 4, 5, 6, 4, 9, 23, 1, 0, 4, 0, 11, 12, 7, 5, 11, 23, 1, 8, 0, 4, 0, 7, 2, 0, 21, 1, 3, 16,
     8, 12, 8, 6, 16, 10, 5, 1, 4, 0, 3, 0, 5, 6, 0, 0, 6, 0, 8, 21, 0, 1, 3, 8, 1, 0, 6, 3, 6, 4, 5, 2, 2, 4, 3, 4, 14,
     2, 1, 17, 4, 1, 17, 4, 0, 0, 5, 22, 20, 2, 2, 6, 1, 11, 7, 3, 19, 3, 2, 22, 2, 1, 6, 11, 0, 7, 2, 6, 5, 15, 7]

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