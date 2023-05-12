import random
import time
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
from config import *
from utils import *
from UAV import UAV
from Bus import Bus
import scienceplots

import argparse

# POI 위치 설정
X, Y, Z = 500, 500, 0
RANDOM_TASK = [{'name':"small",'min':5,'max':15},{'name':"medium",'min':15,'max':25},{'name':"large",'min':25,'max':35},{'name':"very large",'min':35,'max':50}]
SCHEME = ["Game","Matching","Offloading","Local"]
SIMUL_NAME = ["Bus Num","UAV Num","Budget","Scheme","Task Size"]
SAVE_X_NAME = ["Bus","UAV","Budget","Scheme","TaskSize"]
SAVE_Y_NAME = ["overhead","UAV_utility","bus_utility","bus_num"]

NUM_OBJECT = [NUM_BUS,NUM_UAV,BUDGET,len(SCHEME),len(RANDOM_TASK)]
NUM_STEP = [NUM_BUS_STEP, NUM_UAV_STEP, NUM_BUDGET_STEP, len(SCHEME),len(RANDOM_TASK)]
STEP = [BUS_STEP, UAV_STEP, BUDGET_STEP, 1, 1]
X_LABEL = ["Number of buses","Number of UAVs","Budget","Scheme","Task Size"]
Y_LABEL = ["UAV overhead","UAV utility","Bus utility","UAV Bus num"]
LEGEND_LABEL = ["Bus=","UAV=","Budget=","",""]

def simul_value(type,i):
    if type==0:
        return NUM_BUS-i*BUS_STEP
    elif type==1:
        return NUM_UAV-i*UAV_STEP
    elif type==2:
        return BUDGET-i*BUDGET_STEP
    elif type==3:
        return SCHEME[i]
    elif type==4:
        return RANDOM_TASK[i]
    return -1

def mean_without_outliers(lst:list, decision):
    lst.sort()
    l = len(lst)
    l_min = int(l*0.1)
    l_max = int(l*0.9)
    lst = lst[l_min:l_max]
    return round(Average(lst), decision)

if __name__ == "__main__":
    # parsing / default = UAV-Bus task
    parser = argparse.ArgumentParser(description="_")
    parser.add_argument("--x", type=int, default=0, help="x value in graph. range 0~3")
    parser.add_argument("--y", type=int, default=1, help="label in graph, range 0~3")
    args = parser.parse_args()

    print("### SIMULATION START ###")

    # make environment
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

    # for graph
    uav_bus_avg_overhead = [[0 for _ in range(NUM_STEP[args.x])] for _ in range(NUM_STEP[args.y])]
    uav_avg_utility = [[0 for _ in range(NUM_STEP[args.x])] for _ in range(NUM_STEP[args.y])]       
    bus_avg_utility = [[0 for _ in range(NUM_STEP[args.x])] for _ in range(NUM_STEP[args.y])]       
    uav_avg_bus_num = [[0 for _ in range(NUM_STEP[args.x])] for _ in range(NUM_STEP[args.y])]   
	
    buses = deepcopy(buses_original)
    uavs = deepcopy(uavs_original)
    scheme = SCHEME[0]
    budget = BUDGET
    task_range = {'min':TASK_CPU_CYCLE,'max':TASK_CPU_CYCLE}
	# UAV 대수 - 버스의 대수를 점점 줄여나가면서 시뮬레이션
    for i in range(NUM_STEP[args.y]):
        if args.y == 3:
            scheme = SCHEME[i]
        elif args.y == 4:
            task_range = RANDOM_TASK[i]

        if args.x == 0:
            buses = deepcopy(buses_original)
        elif args.x == 1:
            uavs = deepcopy(uavs_original)
        elif args.x == 2:
            budget = BUDGET
        elif args.x == 3:
            pass

        for j in range(NUM_STEP[args.x]):
            print(f"### {SIMUL_NAME[args.y]}:{simul_value(args.y,i)} {SIMUL_NAME[args.x]}:{simul_value(args.x,j)} simulation start")

            # simulate
            simulation(SIMUL_TIME,uavs,buses,scheme=scheme,budget=budget,random_task_range=task_range)
            print("### SIMULATION RESULT ###")

            tmp_overhead = []
            tmp_uav_utility = []
            tmp_bus_utility = []
            tmp_bus_num = []
            
            under = 0
            upper = 0

            for uav in uavs:
                avg_overhead = mean_without_outliers(uav.overhead_list,4)
                avg_utility = mean_without_outliers(uav.utility_list,4)
                avg_bus_num = mean_without_outliers(uav.bus_num_list,4)
                tmp_overhead.append(avg_overhead)
                tmp_uav_utility.append(avg_utility)
                tmp_bus_num.append(avg_bus_num)
                print(f"UAV(ID={uav.id}) has overhead : {avg_overhead}, utility : {avg_utility}")
            
            for bus in buses:
                avg_utility = mean_without_outliers(bus.utility_list,4)
                tmp_bus_utility.append(avg_utility)
                print(f"BUS(ID={bus.id}) has utility : {avg_utility}")

            uav_bus_avg_overhead[i][j] = round(Average(tmp_overhead), 4) #mean_without_outliers(tmp_overhead,4)
            print("over : ",uav_bus_avg_overhead[i][j])
            uav_avg_utility[i][j] = round(Average(tmp_uav_utility), 4) #mean_without_outliers(tmp_uav_utility,4)
            uav_avg_bus_num[i][j] = round(Average(tmp_bus_num), 4) #mean_without_outliers(tmp_bus_num,4)
            bus_avg_utility[i][j] = round(Average(tmp_bus_utility), 4) #mean_without_outliers(tmp_bus_utility,4)

            #print(f"UAV overhead : {AVE}, Under : {under}, Upper : {upper}")
            if args.x == 0:
                for k in range(BUS_STEP):
                    del buses[-1]
            elif args.x == 1:
                for k in range(UAV_STEP):
                    del uavs[-1]
            elif args.x == 2:
                budget -= BUDGET_STEP
            elif args.x == 3:
                scheme = SCHEME[j]
            elif args.x == 4:
                task_range = RANDOM_TASK[j]
            
        if args.y == 0:
            for k in range(BUS_STEP):
                del buses[-1]
        elif args.y == 1:
            for k in range(UAV_STEP):
                del uavs[-1]
        elif args.y == 2:
            budget -= BUDGET_STEP



    # print result

    x_idx = np.arange(0,NUM_STEP[args.x])
    x = NUM_OBJECT[args.x] - x_idx*STEP[args.x]
    legend_value_idx = np.arange(0,NUM_STEP[args.y])
    legend_value = []
    for i in legend_value_idx:
        v = NUM_OBJECT[args.y] - i * STEP[args.y]
        if args.y == 3:
            v = SCHEME[NUM_OBJECT[args.y]-v]
        elif args.y == 4:
            v = RANDOM_TASK[NUM_OBJECT[args.y]-v]['name']
        legend_value.append(v)
    data = [uav_bus_avg_overhead,uav_avg_utility,bus_avg_utility,uav_avg_bus_num]

    plt.style.use(['science','ieee'])

    for i in range(4):
        for j in range(len(legend_value)):
            plt.plot(x, data[i][j], label=LEGEND_LABEL[args.y]+str(legend_value[j]))
            #print(i,j,data[i][j])
        plt.xlabel(X_LABEL[args.x])
        plt.ylabel(Y_LABEL[i])
        plt.legend(loc='upper right')
        #plt.show()
        plt.savefig("./test_graphs/"+SAVE_X_NAME[args.x]+"_"+SAVE_X_NAME[args.y]+"_"+SAVE_Y_NAME[i])
        plt.clf()

    plt.plot(x,data[0][0], label="UAV overhead")
    plt.plot(x,data[1][0], label="UAV utility")
    plt.plot(x,data[2][0], label="Bus utility")

    plt.xlabel(X_LABEL[args.x])
    plt.ylabel("overhead,utility")
    plt.legend(loc='upper right')
    plt.savefig("./test_graphs/"+SAVE_X_NAME[args.x]+"_"+SAVE_X_NAME[args.y]+"_"+"overutil")