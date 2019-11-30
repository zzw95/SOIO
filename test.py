from soio.algo import GWO, CGWO, WOA, CWOA, WOA1, BBO
from soio.util.benchmarks import F1

# problem = F1()
# algorithm = GWO.GreyWolfOptimizer(
#     problem = problem,
#     swarm_size = 100,
#     max_nfes = 10000
# )
# algorithm.run()
# print(algorithm.get_name(), algorithm.get_result().objective)
#
problem = F1()
algorithm = CGWO.CellularGreyWolfOptimizer(
    problem = problem,
    max_nfes = 10000,
    grid_shape=[10, 10],
    neighbor_structure=CGWO.CA_C21,
)
algorithm.run()
print(algorithm.get_name(), algorithm.get_result().objective)
#
# problem = F1()
# algorithm = WOA.WhaleOptimizer(
#     problem = problem,
#     swarm_size = 100,
#     max_nfes = 10000,
# )
# algorithm.run()
# print(algorithm.get_name(), algorithm.get_result().objective)
#
# problem = F1()
# algorithm = WOA1.WhaleOptimizer(
#     problem = problem,
#     swarm_size = 100,
#     max_nfes = 10000,
# )
# algorithm.run()
# print(algorithm.get_name(), algorithm.get_result().objective)
#
# problem = F1()
# algorithm = CWOA.CellularWhaleOptimizer(
#     problem = problem,
#     max_nfes = 10000,
#     grid_shape=[10, 10],
#     neighbor_structure=CWOA.CA_C21,
# )
# algorithm.run()
# print(algorithm.get_name(), algorithm.get_result().objective)

# problem = F1()
# algorithm = BBO.BBO(
#     problem = problem,
#     habitats_size=100,
#     max_nfes = 10000,
# )
# algorithm.run()
# print(algorithm.get_name(), algorithm.get_result().objective)
# print(algorithm.records)
