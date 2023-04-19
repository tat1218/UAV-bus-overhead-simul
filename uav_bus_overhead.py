import random
import time
import math
from copy import deepcopy
import itertools
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

num_bus = 50  # 운행하는 버스의 대수(버스별로 자기 노선(path)을 가짐
num_rsu = 10  # 설치된 RSU의 개수
num_uav = 10  # UAV의 개수
num_path = 200  # 버스 운행경로별로 지나는 정류장(지점)의 개수
num_passenger = 1 # 사용안함
map_size = 1000  # MAP크기
min_height = 100 # UAV의 최저 고도
max_height = 150 # UAV의 최고 고도
poi_radius = 200 # POI로부터 UAV가 위치하는 최대 반경
rsu_distance = 50 # RSU가 설치될 때 서로간의 최소 이격거리
time_interval = 1 
simul_time = 200 # 시뮬레이션을 반복하는 횟수(t)
bus_step = 10 # 버스의 숫자를 변화시키는 횟수(x축)
uav_step = 5 # UAV의 숫자를 변화시키는 횟수
bus_speed = 6.417 # m/s = 23.1km/h / 참고 : https://news.seoul.go.kr/traffic/archives/285
bus_speed_city = 5.333 # m/s = 19.2km/h
task_cpu_cycle = 20 # 단위 TASK 수행에 요구되는 CPU사이클
task_data_size = 20 # 단위 TASK의 파일용량(MB)
task_delay= 10 # 단위 TASK 수행의 최대허용 딜레이(초)

bus_cpu_cycle = 100	# 버스의 최대 cpu 사이클
rsu_cpu_cycle = 100 # RSU의 최대 cpu 사이클
uav_cpu_cycle = 10 # UAV의 최대 cpu 사이클
budget = 50 # UAV의 최대 budget (이 budget을 이용하여 버스나 RSU로부터 cpu를 구매)
sigma_speed = 1000 # 게임이론을 적용해서 버스가 자신의 cpu가격을 변화시킬때 변화값을 판별하기 위한 값

alpha = 0.5 #overhead를 결정할 때, 딜레이와 에너지의 비율

bandwidth = 10e6
bus_energy = 1000 #(사용안함)
uav_energy = 1000 #(사용안함)
uav_computing_unit_energy = 1 # UAV가 local로 task 수행시 사용하는 단위 에너지
uav_transmission_unit_energy = 0.25 # UAV가 bus나 rsu에 데이터전송시 사용하는 단위 에너지
bus_computing_unit_energy = 0.25 # 버스가 task 수행시 사용하는 단위 에너지

no_change_count = 0 # 게임이론으로 자신의 cpu에 대한 price를 변경한 버스의 대수
changed = 1 # 버스로부터 cpu를 구매한 UAV가 있는지를 나타내는 변수
iteration = 1 # 게임이론으로 자신의 cpu에 대한 price를 변경시킨 버스가 존재하는지 여부를 나타내는 변수

# 그래프 smoothing 함수
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

# 시뮬레이션 메인 함수
# 주어진 simul_time 동안 반복해서 수행
# 매 반복시마다 아래의 단계를 수행
# 1단계) 버스의 변수 초기화 및 path상의 다음 위치로 이동, UAV의 변수 초기화
# 2단계) UAV의 통신가능범위에 2대 이상의 버스가 존재하는지 조사(해당 버스들에 대해서만 다음 단계에서 price를 게임이론에 따라 조정)

def simulation(time):

    for simul_i in range(simul_time):

        iteration = 1
        changed = 1
		
		# 버스와 UAV의 인접매트릭스 초기화
        uav_bus_near_matrix = []
        for i in range(num_uav):
            uav_bus_near_matrix.append([])
            for j in range(num_bus):
                uav_bus_near_matrix[i].append(0)

		# UAV가 2대 이상의 버스와 인접한지를 나타내는 인접버스 리스트 초기화
        uav_has_more_than_2bus = []
        for i in range(num_bus):
            uav_has_more_than_2bus.append(0)
		
		# UAV 변수 및 overhead 초기화
        for uav in uavs:
            uav.task_init()
            uav.overhead_init()
		
		# 버스 변수 초기화 및 다음 path로 이동
        for bus in buses:
            bus.init()
            bus.move()
		
		# 버스와 UAV간 전송률 및 딜레이를 구하고, 250m 이내에 위치한 버스와 UAV인 경우 각각 서로의 리스트(bus_list, uav_list)에 추가
		# 버스와 UAV의 인접매트릭스에도 반영
		
        for uav, bus in itertools.product(uavs, buses):
            distance = int (((uav.x - bus.x) ** 2 + (uav.y - bus.y) ** 2 + (uav.z) ** 2) ** 0.5)
            sinr = dbm_to_watt(20 - (131 + 42.8 * math.log10(distance/1000)) + 114)
            transmission_rate = bandwidth * math.log2(1 + sinr) / 1024 / 1024
            transmission_delay = uav.task[1] / transmission_rate
            bus_list = [bus.id, bus.cpu, bus.price, transmission_rate, bus.cpu]
            uav_list = [uav.id, transmission_delay, transmission_rate, 0]

            if distance <= 250 and transmission_rate > 5:
                uav.bus_list.append(bus_list)
                bus.uav_list.append(uav_list)
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

        iter_count = 0

        while(iteration):
            
            iteration = 0
            iter_count +=1

            for i in range(num_bus):
                for j in range(num_uav):
                    bus_uav_list[i][j]=0
			
			
			# 모든 UAV에 대하여 주변의 버스로부터 구입가능한 최대 CPU를 구하는 부분
			
            for uav in uavs:
                summa=0               
                price_num = 0
                uav.BUDGET = budget
                uav.bus_list.sort(key=lambda x: x[2], reverse=True)

                if uav.bus_list: # UAV의 주변에 버스가 존재하면

                    bus_id_list = Extract(uav.bus_list) # 주변 버스의 ID를 bus_id_list에 대입

                    # 게임이론을 적용하기 위한 값 계산
                    for k in range(len(uav.bus_list)):
                        summa += uav.bus_list[k][2]
                        if uav.bus_list[k][2] > 0:
                            price_num += 1

                    for i in range(len(bus_id_list)):

                        T_transmission = uav.task[1] / uav.bus_list[i][3]
                        T_offload = uav.task[0] / uav.bus_list[i][4]
                        E_transmission = T_transmission * uav_transmission_unit_energy
                        E_offload = T_offload * bus_computing_unit_energy * bus_cpu_cycle ** 3

                        busid = bus_id_list[i]
						
						# 게임이론에 따라 UAV가 버스로부터 구입할 수 있는 최대의 CPU사이클 계산
                        MAX_CPU = (uav.BUDGET + 1 * summa) / (buses[busid].price * price_num) - 1
                        
                        if MAX_CPU > 0:
                            bus_uav_list[busid][uav.id] = MAX_CPU
                           
            			
            no_change_count = 0
            num_list = 0

			# 모든 버스에 대하여, UAV의 인접버스리스트에 해당하는 경우에 게임이론을 적용하여 price를 변경
			# 이 과정을 끝내고 나면, 모든 버스는 가장 최적의 price를 설정하게 됨(모든 버스의 초기 price는 1)
			
            for bus in buses:

                if uav_has_more_than_2bus[bus.id] == 1 :

                    num_list = num_list + 1
                    demand_sum_from_uav = 0

                    for i in range(num_uav):
                        demand_sum_from_uav = demand_sum_from_uav + bus_uav_list[bus.id][i]
					
					# 변경된 price 계산
                    temp_price = round(bus.price + (demand_sum_from_uav - bus.MAX_CPU) / sigma_speed,4)

					# 변경된 price와 기존 price의 차이가 threshold 이상인지 검사
                    if abs((temp_price - bus.price)/bus.price) >= (1 / sigma_speed):
						
						# 변경된 price와 기존 price의 차이가 threshold 이상이지만, 2회전 price와 가격차이가 크지 않은지 검사
                        if abs(bus.old_price - temp_price) <= (1 / sigma_speed) * 2:
                            no_change_count += 1
                        
						# 현재 price를 변경
                        else :
                            bus.old_price = bus.price
                            bus.price = temp_price
                            bus.price_history.append(temp_price)
                            iteration = 1
                    
					# price를 변경하지 않은 버스의 대수를 계산
                    else:
                        no_change_count += 1

			# 모든 버스가 더이상 price를 변경하지 않았다면 가격변경 중단
            
            if no_change_count == num_list:
                iteration = 0
			
			# 버스가 price를 바꿔가다가 더이상 바꾸지 않으면 while문을 벗어나야 하는데,
			# price를 계속 바꾸어 무한루프에 빠지는 현상이 있어서, 그걸 벗어나기 위해 iter_count를 사용
			# iter_count가 1000이상이면 price를 더이상 바꾸지 않고 중단		
            
            if iter_count > 1000:
                iteration = 0


        for uav, bus in itertools.product(uavs, buses):
            distance = int (((uav.x - bus.x) ** 2 + (uav.y - bus.y) ** 2 + (uav.z) ** 2) ** 0.5)
            sinr = dbm_to_watt(20 - (131 + 42.8 * math.log10(distance/1000)) + 114)
            transmission_rate = bandwidth * math.log2(1 + sinr) / 1024 / 1024
            transmission_delay = uav.task[1] / transmission_rate
            bus_list = [bus.id, bus.cpu, bus.price, transmission_rate, bus.cpu]
            uav_list = [uav.id, transmission_delay, transmission_rate, 0]

            if distance <= 250 and transmission_rate > 5:
                uav.bus_list.append(bus_list)
                bus.uav_list.append(uav_list)

        for bus in buses:
            bus.uav_list.sort(key=lambda x: x[1])

        for uav in uavs:
            uav.bus_list.sort(key=lambda x: x[2])

        
		# UAV가 버스로부터 실제로 cpu를 구매하는 단계
		# 앞선 버스가 price를 결정하는 부분과 일정부분 동일
		# 더이상 버스로부터 CPU를 구매하는 UAV가 없을 때까지 반복
        
        n_th = 0
        while(changed):
            n_th +=1
            changed = 0

            for uav in uavs:
                summa=0
                price_num = 0
                uav.bus_list.sort(key=lambda x: x[2], reverse=True)

                if uav.bus_list:

                    bus_id_list = Extract(uav.bus_list)

                    for i in range(len(bus_id_list)):

                        for k in range(len(uav.bus_list)):
                            summa += uav.bus_list[k][2]

                        for p in range(len(uav.bus_list)):
                            if uav.bus_list[p][2] > 0 :
                                price_num+=1

                        T_transmission = uav.task[1] / uav.bus_list[i][3]
                        T_offload = uav.task[0] / uav.bus_list[i][4]
                        E_transmission = T_transmission * uav_transmission_unit_energy
                        E_offload = T_offload * bus_computing_unit_energy * bus_cpu_cycle ** 3
						
						# UAV가 버스로부터 CPU를 구매
                        cost = uav.purchase_cpu(i, bus_id_list[i], summa,price_num, T_transmission, T_offload, E_transmission, E_offload, n_th)

                        summa = 0
                        price_num = 0
						
						# CPU를 구매한 버스는 UAV의 리스트로부터 삭제하기 위하여 삭제 리스트에 등록(미리 삭제하면, for문을 돌다가 index범위를 벗어나게 됨)
                        if cost > 0:
                            uav.remove_list.append(i)
			
			# CPU를 구매한 UAV가 존재한다면 반복
            
            for uav in uavs:
                if uav.remove_list:
                    changed = 1
			
			# UAV의 삭제리스트에 등록된 버스를 bus_list로부터 삭제
			
            for uav in uavs:
                temp = len(uav.remove_list)
                temp_bus_list = deepcopy(uav.bus_list)
                temp_uav_remove_list = deepcopy(uav.remove_list)

                for i in range(temp):
                    x = temp_uav_remove_list[i]
                    uav.bus_list.remove(temp_bus_list[x])
                    uav.remove_list.remove(x)
			
			# 버스로부터 CPU를 구매한 UAV의 overhead를 계산
            for uav in uavs:
                uav.overhead_update()

    return 0

def Distance(x1, y1, x2, y2):
    return math.sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2))

def Average(lst):
    return sum(lst) / len(lst)

def Extract(lst):
    return [item[0] for item in lst]

def dbm_to_watt(dbm):
    return 10 ** (dbm / 10) / 1000

def watt_to_dbm(watt):
    return 10 * math.log10(1000 * watt)

class Bus:
    def __init__(self, id, path):
        self.id = id
        self.type = 0      # 0이면 버스, 1이면 RSU
        self.location = [0.0, 0.0]
        self.x = self.location[0]
        self.y = self.location[1]
        self.passengers = []
        self.status = "STOPPED"
        self.path = path
        self.path_index = 0
        self.order = 0
        self.closest = []
        self.uav_list = []
        self.preference_list = []
        self.cpu = bus_cpu_cycle
        self.MAX_CPU = bus_cpu_cycle
        self.sell_uav_list = []
        self.price = 1 # 자신의 여유분 cpu에 대한 cost 부여
        self.old_price = 1
        self.price_history = []
        self.speed = bus_speed if self.type==0 else 0

    def init2(self):
        self.location = [0, 0]
        self.x = self.location[0]
        self.y = self.location[1]
        self.path_index = 0
        self.order = 0

    def init(self):
        self.closest = []
        self.uav_list = []
        self.preference_list = []
        if self.type ==0 :
            self.cpu = bus_cpu_cycle
            self.MAX_CPU = bus_cpu_cycle
        elif self.type ==1:
            self.cpu = rsu_cpu_cycle
            self.MAX_CPU = rsu_cpu_cycle
        self.sell_uav_list = []
        self.price = 1  # 자신의 여유분 cpu에 대한 cost 부여
        self.old_price = 1

    def move(self):
        if self.type == 0:
            move_dist = self.speed
            while move_dist>0:
                self.path_index += 1
                if self.path_index == len(self.path):
                    self.path_index = 0
                p = self.path[self.path_index]
                d = Distance(self.x,self.y,p[0],p[1])
                if d>move_dist:                    
                    self.x = self.x + (p[0]-self.location[0])*(move_dist/d)
                    self.y = self.y + (p[1]-self.location[1])*(move_dist/d)
                    self.location = [self.x,self.y]
                    move_dist = 0
                else:
                    move_dist = move_dist - d
                    self.location = p
                    self.x = self.location[0]
                    self.y = self.location[1]
            # print(f"Bus {self.id} moved to {self.location}")

    def sell_cpu(self, cpu_amount, uav_id):
        self.uav_list.remove(self.uav_list[0])
        if self.cpu >= cpu_amount:
            self.sell_uav_list.append(uav_id)
            self.cpu = round(self.cpu - cpu_amount, 0)
            return cpu_amount * self.price
        else:
            return 0

class RSU:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.closest = []
        self.uav_list = []
        self.preference_list = []
        self.cpu = rsu_cpu_cycle
        self.MAX_CPU = rsu_cpu_cycle
        self.sell_uav_list = []
        self.price = 1  # 자신의 여유분 cpu에 대한 cost 부여
        self.old_price = 0

    def init(self):
        self.closest = []
        self.uav_list = []
        self.preference_list = []
        self.cpu = rsu_cpu_cycle
        self.MAX_CPU = rsu_cpu_cycle
        self.sell_uav_list = []
        self.price = 1  # 자신의 여유분 cpu에 대한 cost 부여
        self.old_price = 0

    def distance(self, other):
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

    def sell_cpu(self, cpu_amount, uav_id):
        self.uav_list.remove(self.uav_list[0])
        if self.cpu >= cpu_amount:
            self.sell_uav_list.append(uav.id)
            self.cpu = round(self.cpu - cpu_amount, 0)
            return cpu_amount * self.price
        else:
            return 0

class UAV:
    def __init__(self, id, x, y, z):
        self.id = id
        self.x = x
        self.y = y
        self.z = z
        self.closest = []
        self.preference_list = []
        #self.task = [round(task_cpu_cycle * random.random(), 2), round(task_data_size * random.random(), 2), round(task_delay * random.random(), 2)]
        self.task = [task_cpu_cycle, task_data_size, task_delay]
        self.task_original = [task_cpu_cycle, task_data_size, task_delay]
        self.cpu = uav_cpu_cycle
        self.energy = uav_energy
        self.budget = budget
        self.BUDGET = budget
        self.T_LOCAL = 0
        self.E_LOCAL = 0
        self.T_local = 0
        self.E_local = 0
        self.T_transmission = 0
        self.T_offload = 0
        self.E_transmission = 0
        self.E_offload = 0
        self.overhead = 0
        self.overhead2 = 0
        self.overhead_list = []
        self.overhead_list2 = []
        self.remove_list = []
        self.buy_cpu = 0  # 구매한 cpu 양 초기화
        self.bus_list = []  # 구매할 버스 리스트 초기화
        self.purchase_bus_list = []  # 구매할 버스 리스트 초기화
        self.cpu_cycle = random.uniform(uav_cpu_cycle*0.5,uav_cpu_cycle)
        self.time_consume = 0

        angle = random.uniform(0, 2 * math.pi)
        distance = random.uniform(0, poi_radius)
        x_offset = distance * math.cos(angle)
        y_offset = distance * math.sin(angle)
        z_offset = random.uniform(min_height, max_height)
        self.x = int(self.x + x_offset)
        self.y = int(self.y + y_offset)
        self.z = int(min(max(self.z + z_offset, min_height), max_height))

    def __str__(self):
        return f"UAV at ({self.x}, {self.y}, {self.z})"

    def overhead_init(self):
        self.T_LOCAL = self.task[0] / self.cpu
        self.E_LOCAL = self.T_LOCAL * uav_computing_unit_energy * self.cpu ** 3
        #self.task_original = self.task.copy()
        self.overhead_list = []

    def overhead_update(self):
        self.T_local = self.task[0] / self.cpu
        self.E_local = self.T_local * uav_computing_unit_energy * self.cpu ** 3
        #self.overhead = alpha * max(self.T_LOCAL, self.T_local+self.T_transmission+self.T_offload) / self.task[2] + (1-alpha) * (self.E_transmission + self.E_local) / self.E_LOCAL
        self.overhead = alpha * ((self.T_local+self.T_transmission+self.T_offload) / self.T_LOCAL) + (1-alpha) * (self.E_transmission + self.E_local) / self.E_LOCAL
        self.overhead_list.append(self.overhead)
        self.overhead_list2.append(self.overhead)
        
    def task_init(self):
        self.closest = []
        self.preference_list = []
        self.task = [task_cpu_cycle, task_data_size, task_delay]
        self.task_original = [task_cpu_cycle, task_data_size, task_delay]
        self.cpu = uav_cpu_cycle
        self.budget = budget
        self.BUDGET = budget
        self.T_LOCAL = 0
        self.E_LOCAL = 0
        self.T_local = 0
        self.E_local = 0
        self.T_transmission = 0
        self.T_offload = 0
        self.E_transmission = 0
        self.E_offload = 0
        self.overhead = 0
        self.overhead2 = 0
        self.remove_list = []
        self.buy_cpu = 0  # 구매한 cpu 양 초기화
        self.bus_list = []  # 구매할 버스 리스트 초기화
        self.purchase_bus_list = []  # 구매할 버스 리스트 초기화
        self.cpu_cycle = random.uniform(uav_cpu_cycle * 0.5, uav_cpu_cycle)
        self.time_consume = 0

    def purchase_cpu(self, i, bus_id_list, sum,num, T_tran, T_off, E_tran, E_off, aaa):
        cost = 0
        buses_sorted = sorted(buses, key=lambda bus: bus.price)
        for bus in buses_sorted:
            uav_id_list = Extract(bus.uav_list)

            if uav_id_list:

                if bus.id == bus_id_list:

                    max_cpu = min(((self.budget+1*sum) / (bus.price*num))-1, bus.cpu, self.budget/bus.price, self.task[0])  # Game theory cpu calculation

                    if max_cpu > 0:
                        cost = bus.sell_cpu(max_cpu, self.id)

                        temp_t = self.time_consume + T_tran + T_off

                        if cost > 0 and temp_t <= self.task[2] :

                            self.time_consume = self.time_consume + temp_t
                            self.buy_cpu += max_cpu
                            self.budget = round(self.budget - cost, 2)
                            self.purchase_bus_list.append([bus.id, round(max_cpu,2), round(max_cpu/bus_cpu_cycle,2)])

                            self.task[0] = self.task[0] - max_cpu
                            self.task[1] = self.task[1] - self.task_original[1] * (max_cpu / self.task_original[0])
                            self.bus_list[i][4] = self.bus_list[i][4] - max_cpu

                            self.T_transmission = self.T_transmission + T_tran
                            self.T_offload = self.T_offload + T_off
                            self.E_transmission = self.E_transmission + E_tran
                            self.E_offload = self.E_offload + E_off

                            bus_uav_list[bus.id][self.id] = [max_cpu]
        return cost

if __name__ == "__main__":
    
    print("### SIMULATION START ###")

    # bus init
    paths = []
    for i in range(num_bus):
        path = [(random.randint(0, map_size), random.randint(0, map_size))]
        while len(path) < num_path:
            x, y = path[-1]
            next_x = random.randint(max(0, x - random.randint(1, 50)), min(map_size, x + random.randint(1, 50)))
            next_y = random.randint(max(0, y - random.randint(1, 50)), min(map_size, y + random.randint(1, 50)))
            if math.dist((x, y), (next_x, next_y)) >= 50:
                path.append((next_x, next_y))
        paths.append(path)

    buses = []
    for i in range(num_bus):
        bus = Bus(i, paths[i])
        buses.append(bus)


    # POI 위치 설정
    #x,y,z = int(random.uniform(0, map_size)), int(random.uniform(0, map_size)), 0
    x, y, z = 500, 500, 0

    # POI로부터 일정거리 이내에 위치하도록 UAV 생성
    uavs = []
    for i in range(num_uav):
        new_uav = UAV(i, x,y,z)
        uavs.append(new_uav)

    bus_diff = int(num_bus / bus_step)
    uav_diff = int(num_uav / uav_step)
    temp_bus_num = num_bus
    temp_uav_num = num_uav
    buses2 = deepcopy(buses)
    uavs2 = deepcopy(uavs)

    AVE = []
    for i in range(uav_step):
        AVE.append([])
        for j in range(bus_step):
            AVE[i].append(0)
	
	# 버스의 대수를 점점 줄여나가면서 시뮬레이션
    
    for k in range(uav_step):
        buses = []
        buses = deepcopy(buses2)
        num_bus = temp_bus_num

        for m in range(bus_step):

            bus_uav_list = []
            for i in range(num_bus):
                bus_uav_list.append([])
                for j in range(num_uav):
                    bus_uav_list[i].append(0)

            for uav in uavs:
                uav.overhead_list2 = []

            simulation(simul_time)

            print("### SIMULATION RESULT ###")

            ave=[]
            for i in range(num_uav):
                ave.append([])

            under = 0
            upper = 0

            for uav in uavs:
                ave[uav.id] = round(Average(uav.overhead_list2), 2)
                print(f"UAV(ID={uav.id}) has overhead : {ave[uav.id]}")
                if ave[uav.id] < 1:
                    under = under + 1
                elif ave[uav.id] >= 1:
                    upper = upper + 1

            AVE[k][m] = round(Average(ave), 2)
            #print(f"UAV overhead : {AVE}, Under : {under}, Upper : {upper}")

            num_bus = num_bus - bus_diff

            for j in range(bus_diff):
                del buses[-1]

            for bus in buses:
                bus.init2()

        num_uav = num_uav - uav_diff
        for j in range(uav_diff):
            del uavs[-1]


    x = np.arange(1,bus_step+1)
    x = np.flip(x)
    x = x * temp_bus_num / bus_step

    plt.plot(x, AVE[0], label='UAV=10')
    plt.plot(x, AVE[1], label='UAV=8')
    plt.plot(x, AVE[2], label='UAV=6')
    plt.plot(x, AVE[3], label='UAV=4')
    plt.plot(x, AVE[4], label='UAV=2')

    plt.gca().invert_xaxis()
    plt.xlabel('# of buses')
    plt.ylabel('UAV overhead')
    plt.legend(loc='upper right')
    plt.show()
