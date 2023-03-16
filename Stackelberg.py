import random


class Bus:
    def __init__(self, id, cpu):
        self.id = id
        self.cpu = cpu
        self.price = None

    def set_price(self, price):
        self.price = price

    def get_price(self):
        return self.price

    def get_cpu(self):
        return self.cpu


class UAV:
    def __init__(self, id, cpu):
        self.id = id
        self.cpu = cpu
        self.bid = None

    def set_bid(self, bid):
        self.bid = bid

    def get_bid(self):
        return self.bid

    def get_cpu(self):
        return self.cpu


# Initialize buses and UAVs
buses = []
uavs = []

for i in range(5):
    bus = Bus(i, random.randint(1, 10))
    buses.append(bus)

for i in range(5):
    uav = UAV(i, random.randint(1, 5))
    uavs.append(uav)

# Leader (Bus) sets prices
for bus in buses:
    prices = []
    for uav in uavs:
        prices.append(uav.get_cpu())
    bus.set_price(max(prices))

# Follower (UAV) chooses bids
for uav in uavs:
    bids = []
    for bus in buses:
        bids.append(bus.get_price())
    uav.set_bid(min(bids))

# Print results
print("Bus CPU: ")
for bus in buses:
    print(f"Bus {bus.id} CPU: {bus.get_cpu()} | Price: {bus.get_price()}")

print("\nUAV Bids: ")
for uav in uavs:
    print(f"UAV {uav.id} CPU: {uav.get_cpu()} | Bid: {uav.get_bid()}")


class Bus:
    def __init__(self, id, cpu):
        self.id = id
        self.cpu = cpu
        self.price = random.uniform(1, 10)  # 자신의 여유분 cpu에 대한 cost 부여

    def sell_cpu(self, cpu_amount):
        # cpu_amount만큼의 cpu를 UAV에게 판매
        if self.cpu >= cpu_amount:
            self.cpu -= cpu_amount
            return cpu_amount * self.price
        else:
            return 0;

class UAV:
    def __init__(self, id, budget):
        self.id = id
        self.budget = int(budget)
        self.cpu = 0  # 구매한 cpu 양 초기화
        self.bus_list = []  # 구매할 버스 리스트 초기화

    def purchase_cpu(self, bus_list):
        # bus_list에서 가장 저렴한 버스를 선택해 cpu 구매
        min_price = None
        min_bus = None

        # for bus in bus_list:
        #     if bus.cpu > 0:  # 여유분 cpu가 있을 때만 고려
        #         # if bus.price < min_price:
        #         # min_price.append(bus.price)
        #         min_bus.append(bus)

        buses_sorted = sorted(bus_list, key=lambda bus: bus.price)

        for bus in buses_sorted:
            max_cpu = min(self.budget // bus.price, bus.cpu)  # budget 내에서 최대한 많은 cpu 구매
            cost = bus.sell_cpu(max_cpu)
            if cost > 0 :
                self.cpu += max_cpu
                self.budget -= cost
                self.bus_list.append(bus.id)

        return cost

    def select_buses(self, bus_list):
        # 버스 리스트를 선택하는 함수. 모든 버스 선택 가능
        self.bus_list = [bus.id for bus in bus_list]


if __name__ == '__main__':
    # 버스와 UAV 초기화
    bus_list = [Bus(i, random.randrange(1, 10)) for i in range(5)]
    uav_list = [UAV(i, random.randrange(1, 10)) for i in range(10)]

    # 버스가 자신의 여유분 cpu에 대한 cost를 부여하는 단계
    for bus in bus_list:
        print(f"Bus {bus.id} price: {bus.price}")

    # UAV가 버스로부터 cpu를 구매하는 단계
    for uav in uav_list:
        uav.purchase_cpu(bus_list)
        print(f"UAV {uav.id}, budget {uav.budget} purchased {uav.cpu} cpu from bus {uav.bus_list}")
