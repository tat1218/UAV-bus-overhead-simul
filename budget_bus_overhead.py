import random
import time
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
from config import *
from utils import *
from UAV import UAV
from Bus import Bus

import argparse

# POI 위치 설정
X, Y, Z = 500, 500, 0

if __name__ == "__main__":
    print("### SIMULATION START ###")
    paths = []
    for i in range(NUM_BUS):
        path = [(random.randint(0, MAP_SIZE), random.randint(0, MAP_SIZE))]
        while len(path) < NUM_PATH:
            x, y = path[-1]
            next_x = random.randint(max(0, x - random.randint(1, 50)), min(MAP_SIZE, x + random.randint(1, 50)))
            next_y = random.randint(max(0, y - random.randint(1, 50)), min(MAP_SIZE, y + random.randint(1, 50)))
            if math.dist((x, y), (next_x, next_y)) >= 50:
                path.append((next_x, next_y))
        paths.append(path)

    buses_original = []
    for i in range(NUM_BUS):
        buses_original.append(Bus(i, 0, paths[i]))

    # POI로부터 일정거리 이내에 위치하도록 UAV 생성
    uavs_original = []
    for i in range(NUM_UAV):
        uavs_original.append(UAV(i, X, Y, Z))

    AVE = [[0 for _ in range(NUM_BUS_STEP)] for _ in range(NUM_UAV_STEP)]
	
    uav_bus_avg_overhead_list = [[0 for _ in range(NUM_BUS_STEP)] for _ in range(NUM_UAV_STEP)]
    uav_avg_utility_list = [[0 for _ in range(NUM_BUS_STEP)] for _ in range(NUM_UAV_STEP)]       
    bus_avg_utility_list = [[0 for _ in range(NUM_BUS_STEP)] for _ in range(NUM_UAV_STEP)]       
	
    uavs = deepcopy(uavs_original)
    budget = BUDGET
	# UAV 대수 - 버스의 대수를 점점 줄여나가면서 시뮬레이션
    for i in range(NUM_BUDGET_STEP):
        buses = deepcopy(buses_original)
        for j in range(NUM_BUS_STEP):
            print(f"### uav budget:{BUDGET-i*BUDGET_STEP} bus num:{NUM_BUS-j*BUS_STEP} simulation start")
            # simulate
            simulation(SIMUL_TIME,uavs,buses,budget=budget)
            print("### SIMULATION RESULT ###")

            tmp_overhead = []
            tmp_uav_utility = []
            tmp_bus_utility = []
            
            under = 0
            upper = 0

            for uav in uavs:
                avg_overhead = round(Average(uav.overhead_list), 2)
                avg_utility = round(Average(uav.utility_list), 2)
                tmp_overhead.append(avg_overhead)
                tmp_uav_utility.append(avg_utility)
                print(f"UAV(ID={uav.id}) has overhead : {avg_overhead}, utility : {avg_utility}")
            
            for bus in buses:
                avg_utility = round(Average(bus.utility_list), 2)
                tmp_bus_utility.append(avg_utility)
                print(f"BUS(ID={bus.id}) has utility : {avg_utility}")

            uav_bus_avg_overhead_list[i][j] = round(Average(tmp_overhead), 2)
            uav_avg_utility_list[i][j] = round(Average(tmp_uav_utility), 2)
            bus_avg_utility_list[i][j] = round(Average(tmp_bus_utility), 2)
            #print(f"UAV overhead : {AVE}, Under : {under}, Upper : {upper}")
            for j in range(BUS_STEP):
                del buses[-1]

        budget -= BUDGET_STEP


    x = np.arange(0,NUM_BUS_STEP)
    x = NUM_BUS - x * BUS_STEP

    # uav - bus overhead
    plt.plot(x, uav_bus_avg_overhead_list[0], label='budget=50')
    plt.plot(x, uav_bus_avg_overhead_list[1], label='budget=40')
    plt.plot(x, uav_bus_avg_overhead_list[2], label='budget=30')
    plt.plot(x, uav_bus_avg_overhead_list[3], label='budget=20')
    plt.plot(x, uav_bus_avg_overhead_list[4], label='budget=10')

    #plt.gca().invert_xaxis()
    plt.xlabel('# of buses')
    plt.ylabel('UAV overhead')
    plt.legend(loc='upper right')
    plt.show()

    # uav utility
    plt.plot(x, uav_avg_utility_list[0], label='budget=50')
    plt.plot(x, uav_avg_utility_list[1], label='budget=40')
    plt.plot(x, uav_avg_utility_list[2], label='budget=30')
    plt.plot(x, uav_avg_utility_list[3], label='budget=20')
    plt.plot(x, uav_avg_utility_list[4], label='budget=10')

    #plt.gca().invert_xaxis()
    plt.xlabel('# of buses')
    plt.ylabel('UAV utility')
    plt.legend(loc='upper right')
    plt.show()

    # bus utility
    plt.plot(x, bus_avg_utility_list[0], label='budget=50')
    plt.plot(x, bus_avg_utility_list[1], label='budget=40')
    plt.plot(x, bus_avg_utility_list[2], label='budget=30')
    plt.plot(x, bus_avg_utility_list[3], label='budget=20')
    plt.plot(x, bus_avg_utility_list[4], label='budget=10')

    #plt.gca().invert_xaxis()
    plt.xlabel('# of buses')
    plt.ylabel('Bus utility')
    plt.legend(loc='upper right')
    plt.show()