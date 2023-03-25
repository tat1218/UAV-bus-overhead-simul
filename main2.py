import random
import time
from math import *
from copy import deepcopy
import itertools

num_bus = 20
num_rsu = 10
num_uav = 10
num_path = 50
num_passenger = 1
map_size = 1000
min_height = 50
max_height = 100
poi_radius = 200
rsu_distance = 50
time_interval = 1
simul_time = 1


uav_cpu = 100
bus_cpu = 100
rsu_cpu = 100
budget = 200
bandwidth = 10e6

bus_cpu_cycle = 5*10e8
uav_cpu_cycle = 10e8
bus_energy = 1000
uav_energy = 1000
uav_computing_unit_energy = 1 / 10e26
uav_transmission_unit_energy = 0.25 / 10e26
bus_computing_unit_energy = 0.25 / 10e26

changed = 1
task_cpu_cycle = 10e9
# The CPU frequency of each vehicle is randomly selected within the range of 0.5–1 GHz. For simplicity, we assume that each vehicle generates zero
# or one task.
# Furthermore, the input data size and number of required CPU cycles for each task are randomly
# selected within the range of 0–1 MB and (0, 1x10^9),
# respectively
task_data_size = 10 # 150~250MB
task_delay= 10

def Extract(lst):
    return [item[0] for item in lst]

def dbm_to_watt(dbm):
    """Convert dBm to Watt"""
    return 10 ** (dbm / 10) / 1000

def watt_to_dbm(watt):
    return 10 * math.log10(1000 * watt)

class Bus:
    def __init__(self, id, path):
        self.id = id
        self.location = [0, 0]
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
        self.cpu = round(bus_cpu * random.random(), 0)
        self.cpu_cycle = bus_cpu_cycle
        self.sell_uav_list = []
        self.price = round(random.uniform(1, 10), 0)  # 자신의 여유분 cpu에 대한 cost 부여
        #self.price = 1  # 자신의 여유분 cpu에 대한 cost 부여

    def move(self):
        if self.location == self.path[self.path_index]:
            self.path_index += 1
            if self.path_index == len(self.path):
                self.path_index = 0
        self.location = self.path[self.path_index]

        self.x = self.location[0]
        self.y = self.location[1]
        # print(f"Bus {self.id} moved to {self.location}")

    def sell_cpu(self, cpu_amount, uav_id):
        self.uav_list.remove(self.uav_list[0])
        if self.cpu >= cpu_amount:
            self.sell_uav_list.append(uav.id)
            self.cpu = round(self.cpu - cpu_amount, 0)
            return cpu_amount * self.price
        else:
            return 0;

    def pick_up(self, passengers):
        for p in passengers:
            self.passengers.append(p)
        # print(f"Bus {self.id} picked up {len(passengers)} passengers")
        self.status = "RUNNING"

    def drop_off(self):
        num_passengerengers = len(self.passengers)
        self.passengers = []
        # print(f"Bus {self.id} dropped off {num_passengerengers} passengers")

    def sort_by_distance(self, uavs):
        self.closest = sorted(uavs,
                              key=lambda uav: ((self.x - uav.x) ** 2 + (self.y - uav.y) ** 2 + (uav.z) ** 2) ** 0.5)


    def run(self):
        self.status = "RUNNING"
        while True:
            self.move()
            time.sleep(time_interval)  # 5초 간격으로 운행

class RSU:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.closest = []
        self.preference_list = []
        self.cpu = rsu_cpu

    def distance(self, other):
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5


class UAV:
    def __init__(self, id, x, y, z):
        self.id = id
        self.x = x
        self.y = y
        self.z = z
        self.closest = []
        self.preference_list = []
        self.task = [round(task_cpu_cycle * random.random(), 0), round(task_data_size * random.random(), 0), round(task_delay * random.random(), 0)]
        self.cpu = round(uav_cpu * random.random(), 0)
        self.energy = uav_energy
        self.budget = budget
        self.remove_list = []
        self.buy_cpu = 0  # 구매한 cpu 양 초기화
        self.bus_list = []  # 구매할 버스 리스트 초기화
        self.purchase_bus_list = []  # 구매할 버스 리스트 초기화
        self.cpu_cycle = random.uniform(uav_cpu_cycle*0.5,uav_cpu_cycle)

        angle = random.uniform(0, 2 * pi)
        distance = random.uniform(0, poi_radius)
        x_offset = distance * cos(angle)
        y_offset = distance * sin(angle)
        z_offset = random.uniform(min_height, max_height)
        self.x = int(self.x + x_offset)
        self.y = int(self.y + y_offset)
        self.z = int(min(max(self.z + z_offset, min_height), max_height))

    def __str__(self):
        return f"UAV at ({self.x}, {self.y}, {self.z})"

    def sort_by_distance(self, buses):
        self.closest = sorted(buses, key=lambda bus: ((self.x - bus.x) ** 2 + (self.y - bus.y) ** 2 + (self.z) ** 2) ** 0.5)

    def purchase_cpu(self, bus_id_list, buses):

        cost = 0
        buses_sorted = sorted(buses, key=lambda bus: bus.price)

        for bus in buses_sorted:

            uav_id_list = Extract(bus.uav_list)

            if uav_id_list:

                #print(f"UAV {self.id}, bus_ld_list {bus_id_list}, i={i}, bus(id)= {bus.id}, uav_id_list[0]={uav_id_list[0]}, uav_id_list={uav_id_list}")

                #if bus.id == i and i == uav_id_list[0]:
                if bus.id == bus_id_list:

                    #offload_size = (self.budget + sum(bus.price for bus in buses_sorted)) / len(range_list) * bus.price -1

                    # UAV가 task를 bus에 offloading 하는 것은 bus로부터 cpu를 얼마만큼 구매하는 개념이 아니라,
                    # 전체 task를 bus에 처리하게 하되, 시간당 cpu 사용값에 대하여 가장 작은 비용을 부가하는 bus를 선택하는 일이 중요
                    # 혹은, 일부분의 task만 보낸다면, 비용을 고려하여 어떤 비율로 나눌(로컬처리, bus 또는 rsu처리) 것인지를 결정할 필요

                    max_cpu = round(min(self.budget // bus.price, bus.cpu),2)  # budget 내에서 최대한 많은 cpu 구매

                    if max_cpu > 0:

                        cost = round(bus.sell_cpu(max_cpu, self.id),2)

                        if cost > 0:
                            self.buy_cpu += max_cpu
                            self.budget = round((self.budget -cost), 2)
                            self.purchase_bus_list.append(bus.id)

                            print(
                                f"UAV {self.id}, budget {self.budget} purchased {self.buy_cpu} cpu from bus {self.purchase_bus_list}")

        return cost


class Passenger:
    def __init__(self, id):
        self.id = id
        self.source = [random.randint(0, map_size), random.randint(0, map_size)]
        self.destination = [random.randint(0, map_size), random.randint(0, map_size)]
        print(f"Passenger {self.id}: source={self.source}, destination={self.destination}")


def calc_sinr_uav_bus(P_tx, lambda_c, d, L, N, B):
    # 송신안테나에서의 송신전력
    P_tx_ant = 10 ** (P_tx / 10) * 1e-3

    # 수신안테나에서의 수신강도
    P_rx_ant = P_tx_ant * (lambda_c / (4 * pi * d)) ** 2 * 10 ** (-L / 10)

    # 총 잡음 (dBm)
    N_total = 10 ** (N / 10) * B

    # SINR 계산
    sinr = 10 * log10(P_rx_ant / N_total)

    return sinr


if __name__ == "__main__":
    import time
    random.seed(time.time())

    paths = []
    for i in range(num_bus):
        path = [(random.randint(0, map_size), random.randint(0, map_size))]
        while len(path) < num_path:
            x, y = path[-1]

            next_x = random.randint(max(0, x - random.randint(1, 50)), min(map_size, x + random.randint(1, 50)))
            next_y = random.randint(max(0, y - random.randint(1, 50)), min(map_size, y + random.randint(1, 50)))

            if dist((x, y), (next_x, next_y)) >= 50:
                path.append((next_x, next_y))
        paths.append(path)

    buses = []
    for i in range(num_bus):
        bus = Bus(i, paths[i])
        buses.append(bus)

    rsus = []
    while len(rsus) < num_rsu:
        x, y = int(random.uniform(0, map_size)), int(random.uniform(0, map_size))
        new_rsu = RSU(x, y)

        if all(new_rsu.distance(existing_rsu) >= rsu_distance for existing_rsu in rsus):
            rsus.append(new_rsu)

    x,y,z = int(random.uniform(0, map_size)), int(random.uniform(0, map_size)), 0

    uavs = []
    for i in range(num_uav):
        new_uav = UAV(i, x,y,z)
        uavs.append(new_uav)

    passengers = []
    for i in range(num_passenger):
        passengers.append(Passenger(i))

    for i in range(simul_time):

        for bus in buses:
            if bus.status == "STOPPED":
                nearby_passengers = [p for p in passengers if abs(p.source[0] - bus.x) <= 50 and abs(
                    p.source[1] - bus.y) <= 50]
                if len(nearby_passengers) > 0:
                    bus.pick_up(nearby_passengers)
                    bus.status = "RUNNING"
                else:
                    bus.move()

            elif bus.status == "RUNNING":
                nearby_passengers = [p for p in bus.passengers if abs(p.destination[0] - bus.x) <= 50 and abs(
                    p.destination[1] - bus.y) <= 50]
                if len(nearby_passengers) > 0:
                    bus.drop_off()
                    bus.pick_up(nearby_passengers)
                    bus.status = "STOPPED"
                else:
                    bus.move()
            else:
                bus.move()

        for uav, bus in itertools.product(uavs, buses):
            distance = int (((uav.x - bus.x) ** 2 + (uav.y - bus.y) ** 2 + (uav.z) ** 2) ** 0.5)

            # freq_m = (3 * 10**8) / (2 * 10**9)
            # sinr = dbm_to_watt(calc_sinr_uav_bus(30, 2*(10**9), distance, 0, -114, 10e6))

            # sinr = dbm_to_watt(30) * ((distance) ** (-3.4)) / dbm_to_watt(-114)
            sinr = dbm_to_watt(20 - (131 + 42.8 * log10(distance/1000)) + 114)
            transmission_rate = round(bandwidth * log2(1 + sinr) / 1024 / 1024, 2)

            transmission_delay = round(uav.task[1] / transmission_rate, 2)

            bus_list = [bus.id, bus.cpu, bus.price, transmission_rate, bus.cpu_cycle]
            uav_list = [uav.id, transmission_delay, transmission_rate, 0]


            if distance <= 250 and transmission_rate > 1:
                uav.bus_list.append(bus_list)
                bus.uav_list.append(uav_list)

        # 버스가 자신의 여유분 cpu에 대한 cost를 부여하는 단계
        for bus in buses:
            bus.uav_list.sort(key=lambda x: x[1])
            print(f"BUS(ID={bus.id}) CPU: {bus.cpu} price : {bus.price} at ({bus.x}, {bus.y}) has the following closest uavs: {bus.uav_list}")

        for uav in uavs:
            uav.bus_list.sort(key=lambda x: x[2])
            print(f"UAV(ID={uav.id}) has the following closest buses: {uav.bus_list}")

        # UAV가 가장 선호하는 버스로부터 cpu를 구매하는 단계
        while(changed):

            changed = 0
            for uav in uavs:

                uav.bus_list.sort(key=lambda x: x[2], reverse=True)

                if uav.bus_list:
                    #print(uav.id, uav.bus_list)
                    bus_id_list = Extract(uav.bus_list)
                    temp_bus_list = deepcopy(uav.bus_list)

                    for i in range(len(bus_id_list)):

                        # offloading 할 때와 local 처리할때의 시간과 에너지 소비 비교
                        T_local = round(uav.task[0] / uav.cpu_cycle, 2)
                        E_local = round(T_local * uav_computing_unit_energy *  uav.cpu_cycle ** 3, 2)
                        T_transmission = round(uav.task[1] / temp_bus_list[i][3], 2)
                        T_offload = round(uav.task[0] / temp_bus_list[i][4], 2)
                        E_transmission = round(T_transmission * uav_transmission_unit_energy, 2)
                        E_offload = round(T_offload * bus_computing_unit_energy * bus_cpu_cycle ** 3, 2)

                        cost = uav.purchase_cpu(bus_id_list[i], buses)

                        if cost > 0:
                            uav.remove_list.append(i)

            for uav in uavs:
                temp = len(uav.remove_list)
                temp_bus_list = deepcopy(uav.bus_list)
                for i in range(temp):
                    x = uav.remove_list[i]

                    uav.bus_list.remove(temp_bus_list[x])

                if uav.bus_list:
                    changed = 1



            #changed = 0

        print("end")

        time.sleep(time_interval)
