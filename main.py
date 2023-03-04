import random
import time
from math import *
import itertools

num_bus = 50
num_rsu = 50
num_uav = 20
num_path = 100
num_passenger = 1
map_size = 1000
min_height = 100
max_height = 150
poi_radius = 50
rsu_distance = 50
time_interval = 1
simul_time = 5

uav_energy = 100
uav_cpu = 100
bus_cpu = 100
rsu_cpu = 100

bandwidth = 10e6


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
        self.closest = []
        self.uav_list = []
        self.preference_list = []
        self.cpu = round(bus_cpu * random.random(), 2)

    def move(self):
        if self.location == self.path[self.path_index]:
            self.path_index += 1
            if self.path_index == len(self.path):
                self.path_index = 0
        self.location = self.path[self.path_index]

        self.x = self.location[0]
        self.y = self.location[1]
        # print(f"Bus {self.id} moved to {self.location}")

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
        self.bus_list = []
        self.preference_list = []
        self.task = [random.randint(1, 100), random.randint(1, 100), random.randint(1, 10)]
        self.cpu = uav_cpu
        self.energy = uav_energy


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

            bus_list = bus.id, bus.cpu
            uav_list = uav.id, transmission_delay, transmission_rate

            if transmission_rate > 1:
                uav.bus_list.append(bus_list)
                bus.uav_list.append(uav_list)

            # av.bus_list.append(bus_list)
            # bus.uav_list.append(uav_list)

            # if distance <=250:
                # print(f"distnace {distance}m, UAV ({uav.x}, {uav.y}, {uav.z}) and Bus ({bus.x}, {bus.y}) can communicate with {transmission_rate} Mbps ")

        for uav in uavs:
            uav.bus_list.sort(key=lambda x: x[1], reverse=True)

            print(
                f"UAV(ID={uav.id}) at ({uav.x}, {uav.y}, {uav.z}) has the following closest buses: {uav.bus_list}")

        for bus in buses:
            bus.uav_list.sort(key=lambda x: x[1])

            print(
                f"BUS(ID={bus.id}) at ({bus.x}, {bus.y}) has the following closest uavs: {bus.uav_list}")

        for uav in uavs:
            uav.bus_list =[]

        for bus in buses:
            bus.uav_list =[]

        time.sleep(time_interval)
