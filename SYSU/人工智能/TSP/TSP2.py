import multiprocessing
import random
import time
import numpy as np

from multiprocessing import Pool


class GeneticAlgTSP:
    def __init__(self, filename):
        # 读取城市、初始化种群、种群大小等
        self.name = filename.split('/')[2].split(".")[0]
        self.cities = self.load_cities(filename)
        self.population_size = 2
        self.population = self.init_population(self.population_size)
        self.dis = []
        self.depth = 3

    # 加载城市坐标
    def load_cities(self, filename):
        cities = []
        start = False
        with open(filename, 'r') as file:
            for line in file.readlines():
                if line == 'NODE_COORD_SECTION\n' and start is False:
                    start = True
                    continue

                parts = line.strip().split()
                if len(parts) == 3 and start:
                    _, x, y = parts
                    cities.append((float(x), float(y)))
        return np.array(cities)

# 初始化种群，给种群添加个体
    def init_population(self, population_size):
        n = len(self.cities)
        population = []
        for _ in range(population_size):
            current_population = random.sample(range(1, n + 1), n)
            population.append(current_population)
        return population

# 两个父代交叉产生子代
    def crossover(self, p1, p2):
        # 处理链式映射
        def mapping(seq, mapping, s, t):
            new = seq
            for i in range(len(seq)):
                if not (s <= i <= t):
                    while new[i] in mapping.values():
                        # 查找链式映射直到找到一个不在交换段内的值
                        for k, v in mapping.items():
                            if v == new[i]:
                                new[i] = k
                                break
            return new
        
        length = len(p1)
        # 随机生成两个在0~length-1范围的下标s,t，确保s<t
        s, t = sorted(random.sample(range(length), 2))

        # 交换p1,p2在s~t的部分
        new_p1 = p1[:s] + p2[s:t + 1] + p1[t + 1:]
        new_p2 = p2[:s] + p1[s:t + 1] + p2[t + 1:]

        # 存储映射关系
        p1_p2 = {}
        for i in range(s, t + 1):
            p1_p2[p1[i]] = p2[i]


        new_p1 = mapping(new_p1, p1_p2, s, t)
        # 反转映射关系，用于第二个子代
        p2_p1 = {v: k for k, v in p1_p2.items()}
        new_p2 = mapping(new_p2, p2_p1, s, t)

        return new_p1, new_p2

    def mutation(self, individual):
        length = len(individual)
        # 随机生成两个在0~length-1范围的下标s,t，确保s<t
        s, t = sorted(random.sample(range(length), 2))
        # 将s,t中间部分倒置
        individual[s:t + 1] = individual[s:t + 1][::-1]
        return individual

    def crossover_and_mutation(self, parents):
        # 交叉
        child1, child2 = self.crossover(parents[0].copy(), parents[1].copy())
        # 倒置变异
        child1 = self.mutation(child1)
        child2 = self.mutation(child2)
        # 交换变异
        child1 = self.mutation(child1)
        child2 = self.mutation(child2)

        return child1, child2

# 计算距离
    def distance(self, individual):
        total = 0
        for i in range(len(individual)):
            start = individual[i - 1]
            end = individual[i]
            d = np.linalg.norm(self.cities[start - 1] - self.cities[end - 1])
            total += d
        return total

# 计算适应度
    def fitness(self, individual):
        total = self.distance(individual)
        return 1 / total

    def select_parents(self, population):
        # 排序
        # return population[0], population[1]

        # 轮盘赌
        values = []

        for individual in population:
            f = self.fitness(individual)  # 调用计算适应度的函数
            values.append(f)

        total_fitness = sum(values)

        selection_probs = []

        # 计算每个适应度值的概率
        for fitness in values:
            prob = fitness / total_fitness
            selection_probs.append(prob)

        p1, p2 = np.random.choice(len(population), 2, p=selection_probs)
        return population[p1], population[p2]

    def select_best_solution(self):
        best_fitness = float('-inf')
        best_individual = None
        for individual in self.population:
            fitness = self.fitness(individual)
            if fitness > best_fitness:
                best_fitness = fitness
                best_individual = individual
        return best_individual



    def iterate(self, num_iterations):
        best_dist = float('inf')
        stag_count = 0

        with Pool(processes=multiprocessing.cpu_count()) as pool:
            iteration = 0
            #for iteration in range(num_iterations):
            while iteration < num_iterations:
                # 选择父母并进行交叉变异
                pairs = [self.select_parents(self.population) for _ in range(len(self.population) // 2)]
                children = pool.map(self.crossover_and_mutation, pairs)
                children = [child for pair in children for child in pair]  # 展平列表

                # 更新种群
                self.population += children
                self.population.sort(key=self.fitness, reverse=True)
                self.population = self.population[:self.population_size]

                sol = self.select_best_solution()
                dist = self.distance(sol)

                # 检查是否更新了最优解
                if dist < best_dist:
                    best_dist = dist
                    stag_count = 0
                    self.dis.append(dist)
                else:
                    stag_count += 1
                    self.dis.append(dist)

                # 扩大搜索范围
                if len(self.cities) * 20 > stag_count >= len(self.cities) * 10 and self.depth:
                    self.depth -= 1
                    # 扩大种群数量
                    self.population_size *= 2
                    self.population += self.init_population(self.population_size - len(self.population))
                    # 头部变异
                    for i in range(1, self.population_size):
                        self.population[i] = self.mutation(self.population[i])

                    print(f'Population size doubled to: {self.population_size}')
                    print(f'Depth: {3 - self.depth}')

                    stag_count = 0

                print(f'Iteration: {iteration + 1}, '
                             f'Best Distance: {dist}')
                iteration += 1
            return sol, best_dist


def main():
    start = time.time()
    ga_tsp = GeneticAlgTSP("./data/wi29.tsp")
    best_solution, best_distance = ga_tsp.iterate(len(ga_tsp.cities) * 100)
    end = time.time()
    print('---- FINAL RESULTS ----')
    print(f'Best Solution: {best_solution}')
    print(f'Best Distance: {best_distance}')
    print(f'Time: {end - start}')


if __name__ == '__main__':
    main()