import numpy as np
import math

# 그래프 smoothing 함수
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def Average(lst):
    return sum(lst) / len(lst)

def Extract(lst):
    return [item[0] for item in lst]

def dbm_to_watt(dbm):
    return 10 ** (dbm / 10) / 1000

def watt_to_dbm(watt):
    return 10 * math.log10(1000 * watt)

from config import *
import itertools
import Bus, UAV
from copy import deepcopy

MAX_ITER_COUNT = 1000

def simulation(simul_time, uavs:list[UAV.UAV], buses:list[Bus.Bus], scheme="Game", budget=BUDGET):
    # 시뮬레이션 메인 함수
    # 주어진 simul_time 동안 반복해서 수행
    # 매 반복시마다 아래의 단계를 수행
    # 1단계) 버스의 변수 초기화 및 path상의 다음 위치로 이동, UAV의 변수 초기화
    # 2단계) UAV의 통신가능범위에 2대 이상의 버스가 존재하는지 조사(해당 버스들에 대해서만 다음 단계에서 price를 게임이론에 따라 조정)

    #setup variables
    num_bus = len(buses)
    num_uav = len(uavs)
    transmission_rate = [[0 for _ in range(num_bus)] for _ in range(num_uav)]
    uav_bus_maxcpu_list = [[0 for _ in range(num_bus)] for _ in range(num_uav)]

    # reset before simulation
    for uav in uavs:
        uav.reset()

    for bus in buses:
        bus.reset()

    # simulate
    for simul_i in range(simul_time):

        iteration = 1
        changed = 1
		
		# 버스와 UAV의 인접매트릭스 초기화
        uav_bus_near_matrix = [[0 for _ in range(num_bus)] for _ in range(num_uav)]

		# UAV가 2대 이상의 버스와 인접한지를 나타내는 인접버스 리스트 초기화
        uav_has_more_than_2bus = [0 for _ in range(num_bus)]
		
		# UAV, 버스 초기화 & 이동
        for uav in uavs:
            uav.init(budget)
		
        for bus in buses:
            bus.init()
            bus.move()
		
		# 버스와 UAV간 전송률 및 딜레이를 구하고, 250m 이내에 위치한 버스와 UAV인 경우 각각 서로의 리스트(bus_list, uav_list)에 추가
		# 버스와 UAV의 인접매트릭스에도 반영
        for uav, bus in itertools.product(uavs, buses):
            distance = int (((uav.x - bus.x) ** 2 + (uav.y - bus.y) ** 2 + (uav.z) ** 2) ** 0.5)
            sinr = dbm_to_watt(20 - (131 + 42.8 * math.log10(distance/1000)) + 114)
            transmission_rate[uav.id][bus.id] = BANDWIDTH * math.log2(1 + sinr) / 1024 / 1024

            if distance <= 250 and transmission_rate[uav.id][bus.id] > 5:
                uav.bus_id_list.append(bus.id)
                bus.uav_id_list.append(uav.id)
                uav_bus_near_matrix[uav.id][bus.id]=1
		
		# 인접매트릭스를 이용하여 인접버스 리스트 생성
        for i in range(num_uav):
            temp2 = 0
            for j in range(num_bus):
                temp2 = temp2 + uav_bus_near_matrix[i][j]

            if temp2 > 1:
                for j in range(num_bus):
                    if uav_bus_near_matrix[i][j] > 0:
                        uav_has_more_than_2bus[j] = 1

        if scheme == "Game":        # Computing Offloading Scheme based on Stackelberg Game
            iter_count = 0
            # 가격 설정 단계
            while(iteration):
                iteration = 0
                iter_count +=1

                for i in range(num_uav):
                    for j in range(num_bus):
                        uav_bus_maxcpu_list[i][j]=0
                
                
                # 모든 UAV에 대하여 주변의 버스로부터 구입가능한 최대 CPU를 구하는 부분
                for uav in uavs:
                    price_sum = 0               
                    price_num = 0

                    if uav.bus_id_list: # UAV의 주변에 버스가 존재하면
                        # 게임이론을 적용하기 위한 값 계산
                        for bus_id in uav.bus_id_list:
                            price_sum += buses[bus_id].price
                            if buses[bus_id].price > 0:
                                price_num += 1

                        for bus_id in uav.bus_id_list:
                            # 게임이론에 따라 UAV가 버스로부터 구입할 수 있는 최대의 CPU사이클 계산
                            MAX_CPU = (BUDGET + 1 * price_sum) / (buses[bus_id].price * price_num) - 1
                            if MAX_CPU > 0:
                                uav_bus_maxcpu_list[uav.id][bus_id] = MAX_CPU
                            
                no_change_count = 0
                num_list = 0

                # 모든 버스에 대하여, UAV의 인접버스리스트에 해당하는 경우에 게임이론을 적용하여 price를 변경
                # 이 과정을 끝내고 나면, 모든 버스는 가장 최적의 price를 설정하게 됨(모든 버스의 초기 price는 1)
                
                for bus in buses:
                    if uav_has_more_than_2bus[bus.id] == 1 :

                        num_list += 1
                        demand_sum_from_uav = 0

                        for i in range(num_uav):
                            demand_sum_from_uav += uav_bus_maxcpu_list[i][bus.id]
                        
                        # 변경된 price 계산
                        temp_price = round(bus.price + (demand_sum_from_uav - bus.MAX_CPU) / SIGMA_SPEED,4)

                        # 변경된 price와 기존 price의 차이가 threshold 이상인지 검사
                        if abs((temp_price - bus.price)/bus.price) >= (1 / SIGMA_SPEED):
                            
                            # 변경된 price와 기존 price의 차이가 threshold 이상이지만, 2회전 price와 가격차이가 크지 않은지 검사
                            if abs(bus.old_price - temp_price) <= (1 / SIGMA_SPEED) * 2:
                                no_change_count += 1
                            
                            # 현재 price를 변경
                            else :
                                bus.change_price(temp_price)
                                iteration = 1
                        
                        # price를 변경하지 않은 버스의 대수를 계산
                        else:
                            no_change_count += 1

                # 모든 버스가 더이상 price를 변경하지 않았다면 가격변경 중단
                
                if no_change_count == num_list:
                    iteration = 0
                
                # 버스가 price를 바꿔가다가 더이상 바꾸지 않으면 while문을 벗어나야 하는데,
                # price를 계속 바꾸어 무한루프에 빠지는 현상이 있어서, 그걸 벗어나기 위해 iter_count를 사용
                # iter_count가 MAX_ITER_COUNT이상이면 price를 더이상 바꾸지 않고 중단		
                
                if iter_count > MAX_ITER_COUNT:
                    iteration = 0
            
            # UAV가 버스로부터 실제로 cpu를 구매하는 단계
            # 앞선 버스가 price를 결정하는 부분과 일정부분 동일
            # 더이상 버스로부터 CPU를 구매하는 UAV가 없을 때까지 반복
            n_th = 0
            while(changed):
                n_th +=1
                changed = 0

                for uav in uavs:
                    price_sum = 0
                    price_num = 0
                    uav.bus_id_list.sort(key=lambda x: buses[x].price, reverse=True)

                    if uav.bus_id_list:
                        for bus_id in uav.bus_id_list:
                            price_sum += buses[bus_id].price
                            if buses[bus_id].price > 0:
                                price_num += 1

                        tmp_list = deepcopy(uav.bus_id_list)
                        for bus_id in tmp_list:
                            # UAV가 버스로부터 CPU를 구매
                            # CPU를 구매한 UAV가 존재한다면 반복
                            if buses[bus_id].cpu > 0 and uav.purchase_cpu(buses[bus_id], transmission_rate[uav.id][bus_id], price_sum, price_num):
                                changed = 1
        elif scheme == "Matching":  # Matching Scheme
            pass
        elif scheme == "Local":     # Local Scheme
            pass
        elif scheme == "Offloading":# Offloading Scheme
            pass
        # 버스의 utility 계산 & 리스트에 추가
        for bus in buses:
            bus.result_update()
		# UAV의 overhead와 utility를 계산 & 리스트에 추가
        for uav in uavs:
            uav.result_update()
