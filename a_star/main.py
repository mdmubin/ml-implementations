from math import sqrt, inf
from heapq import heappush, heappop


def get_heuristic(node_data, destination):
    """
    DESCRIPTION
    -----------

    Returns a mapping of all the heuristic estimates of reaching the
    destination node from every other nodes in the `node_data` list by using
    their Euclidean distances from the goal node.

    PARAMETERS
    ----------

    - node_data [ type: `dict [ NodeType : tuple(float, float) ]` ]:
        Expects a dictionary containing all the nodes mapped to their cartesian
        positions, i.e. nodes are mapped to their  x and y values

    - destination [ type: `NodeType` ]:
        The destination node. Heuristic values will be generated for every node
        in `node_data` relative to this node.

    RETURNS
    -------

    - type: `dict [ NodeType : float ]`

    A dictionary mapping the nodes to their distances in the format
    `node -> h(node)`.
    """

    def _euclid_dist(pos1, pos2):
        # √( (p1.x - p2.x)^2 + (p1.y - p2.y)^2 )
        return sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    heuristic_map = {}
    for node in node_data:
        heuristic_map[node] = _euclid_dist(node_data[node],
                                           node_data[destination])
    return heuristic_map


def a_star_path(graph, start, end, node_data):
    """
    DESCRIPTION
    -----------

    Generates a path and the path's cost from the start to the end node of a
    graph by using the A-star search algorithm. 
    
    * Reference: https://en.wikipedia.org/wiki/A*_search_algorithm

    PARAMETERS
    ----------

    - graph [ type: `dict[ NodeType -> list [ NodeType, Float ] ]` ]:
        The graph data structure on which the algorithm will run. The graph is
        expected to be similar to a `dict` mapping nodes to a list of its
        adjacent nodes

    - start [ type: `NodeType` ]:
        The starting point/node of the path within the graph

    - end [ type: `NodeType` ]
        The destination point/node of the path within the graph

    - node_data [ type: `dict [ NodeType : tuple(float, float) ]` ]:
        Expects a dictionary containing all the nodes mapped to their cartesian
        positions, i.e. nodes are mapped to their  x and y values

    RAISES
    ------

    - `KeyError`:
        If the end node is unreachable from the start node.

    RETURNS
    -------

    - output format: `dict[ "path" : list[NodeType], "cost" : float ]`

    a dict containing the path and cost of the path from the start to the end
    node using a-star search.
    """

    h = get_heuristic(node_data, end)  # heuristic maps

    heap = []  # min heap to track next nodes
    heappush(heap, (0, start))

    # map containing true cost to reach each node from source
    cost = {}
    for node in graph:
        # Mark all nodes as unreachable from parent node
        cost[node] = inf
    # starting node is always reachable
    cost[start] = 0

    # no nodes have parents initially, i.e. parent[node] = None
    parent = dict.fromkeys(graph.keys(), None)

    while heap:
        cur_cost, cur_node = heappop(heap)

        # if we have reached end, return cost and path
        if cur_node == end:
            # build path
            path = []
            n = end
            while parent[n]:
                path.append(n)
                n = parent[n]
            path.append(start)
            path.reverse()
            # return path and cost
            return {'path': path, 'cost': cur_cost}

        # for each adjacent node to current node
        for adj, w in graph[cur_node]:
            # total estimated cost = cost to reach this node + heuristic value
            new_cost = cost[cur_node] + w
            # if new cost is lower than previous true cost, update prev values
            if new_cost < cost[adj]:
                cost[adj] = new_cost
                parent[adj] = cur_node

                estimated_cost = new_cost + h[adj]
                heappush(heap, (estimated_cost, adj))

    raise KeyError(f'destination node is unreachable from source node')


if __name__ == '__main__':
    # Reading file input
    with open('a_star/input.txt', 'r') as graph_in:
        file_input = [lines.strip() for lines in graph_in.readlines()]

    # Getting node data
    # node data format = { node : (nodeX, nodeY) }
    node_count = int(file_input[0])
    node_data = {}

    for i in range(1, node_count + 1):
        node_info = file_input[i].split(' ')
        node_data[node_info[0]] = (int(node_info[1]), int(node_info[2]))

    # Creating graph using node list
    # graph format = { node -> [ [neighbor, weight] , ...] }
    graph = {}

    for node in node_data:
        graph[node] = []

    # Add edges between nodes using file data
    # NOTE: data offset used to extract information from file_input
    edge_count = int(file_input[node_count + 1])

    for i in range(node_count + 2, node_count + edge_count + 2):
        edge_start, edge_end, weight = file_input[i].split()
        # graph[node] -> [ [neighbor, weight], ... ]
        graph[edge_start].append([edge_end, int(weight)])

    # get source and goal nodes
    source, goal = file_input[-2], file_input[-1]
    print(a_star_path(graph, source, goal, node_data))
