"""
AI-DRIVEN PATHFINDING VISUALIZER - FLASK BACKEND
Design and Analysis of Algorithms (DAA) - B.Tech CSE 5th Semester
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import json
from queue import PriorityQueue
import heapq
import time
from collections import deque

app = Flask(__name__)
CORS(app)

# ============================================================================
# PATHFINDING ALGORITHMS
# ============================================================================

class PathfindingVisualizer:
    def __init__(self, grid):
        """Initialize the visualizer with a grid"""
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if grid else 0
        self.visited = []
        self.path = []
        self.distances = {}
        self.execution_time = 0
        
    def get_neighbors(self, pos):
        """Get valid neighbors (4-directional: up, down, left, right)"""
        x, y = pos
        neighbors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.rows and 0 <= ny < self.cols:
                if self.grid[nx][ny] != 1:  # 1 = wall
                    neighbors.append((nx, ny))
        return neighbors
    
    def reconstruct_path(self, parent, end):
        """Reconstruct path from parent pointers"""
        path = []
        current = end
        while current in parent:
            path.append(current)
            current = parent[current]
        path.reverse()
        return path
    
    def dijkstra(self, start, end):
        """Dijkstra's Algorithm"""
        start_time = time.time()
        
        distance = {(i, j): float('inf') for i in range(self.rows) 
                   for j in range(self.cols) if self.grid[i][j] != 1}
        distance[start] = 0
        parent = {}
        visited = set()
        pq = [(0, start)]
        visited_order = []
        
        while pq:
            curr_dist, current = heapq.heappop(pq)
            
            if current in visited:
                continue
                
            visited.add(current)
            visited_order.append(current)
            
            if current == end:
                path = self.reconstruct_path(parent, end)
                self.execution_time = time.time() - start_time
                self.visited = visited_order
                self.path = path
                return {
                    'found': True,
                    'path': path,
                    'visited': visited_order,
                    'distance': distance[end],
                    'cells_visited': len(visited_order),
                    'time': self.execution_time,
                    'algorithm': 'Dijkstra'
                }
            
            for neighbor in self.get_neighbors(current):
                new_dist = distance[current] + 1
                if new_dist < distance[neighbor]:
                    distance[neighbor] = new_dist
                    parent[neighbor] = current
                    heapq.heappush(pq, (new_dist, neighbor))
        
        self.execution_time = time.time() - start_time
        self.visited = visited_order
        return {
            'found': False,
            'visited': visited_order,
            'cells_visited': len(visited_order),
            'time': self.execution_time,
            'algorithm': 'Dijkstra'
        }
    
    def bfs(self, start, end):
        """Breadth-First Search"""
        start_time = time.time()
        
        visited = set([start])
        queue = deque([start])
        parent = {}
        visited_order = [start]
        
        while queue:
            current = queue.popleft()
            
            if current == end:
                path = self.reconstruct_path(parent, end)
                self.execution_time = time.time() - start_time
                self.visited = visited_order
                self.path = path
                return {
                    'found': True,
                    'path': path,
                    'visited': visited_order,
                    'distance': len(path) - 1,
                    'cells_visited': len(visited_order),
                    'time': self.execution_time,
                    'algorithm': 'BFS'
                }
            
            for neighbor in self.get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = current
                    queue.append(neighbor)
                    visited_order.append(neighbor)
        
        self.execution_time = time.time() - start_time
        self.visited = visited_order
        return {
            'found': False,
            'visited': visited_order,
            'cells_visited': len(visited_order),
            'time': self.execution_time,
            'algorithm': 'BFS'
        }
    
    def dfs(self, start, end):
        """Depth-First Search"""
        start_time = time.time()
        
        visited = set()
        parent = {}
        visited_order = []
        
        def dfs_recursive(current):
            visited.add(current)
            visited_order.append(current)
            
            if current == end:
                return True
            
            for neighbor in self.get_neighbors(current):
                if neighbor not in visited:
                    parent[neighbor] = current
                    if dfs_recursive(neighbor):
                        return True
            
            return False
        
        found = dfs_recursive(start)
        
        if found:
            path = self.reconstruct_path(parent, end)
            self.execution_time = time.time() - start_time
            self.visited = visited_order
            self.path = path
            return {
                'found': True,
                'path': path,
                'visited': visited_order,
                'distance': len(path) - 1,
                'cells_visited': len(visited_order),
                'time': self.execution_time,
                'algorithm': 'DFS'
            }
        
        self.execution_time = time.time() - start_time
        self.visited = visited_order
        return {
            'found': False,
            'visited': visited_order,
            'cells_visited': len(visited_order),
            'time': self.execution_time,
            'algorithm': 'DFS'
        }
    
    def a_star(self, start, end):
        """A* Algorithm with Manhattan distance heuristic"""
        start_time = time.time()
        
        def heuristic(pos):
            return abs(pos[0] - end[0]) + abs(pos[1] - end[1])
        
        distance = {(i, j): float('inf') for i in range(self.rows) 
                   for j in range(self.cols) if self.grid[i][j] != 1}
        distance[start] = 0
        parent = {}
        visited = set()
        visited_order = []
        
        # Priority queue: (f_score, counter, node)
        pq = [(heuristic(start), 0, start)]
        counter = 1
        
        while pq:
            _, _, current = heapq.heappop(pq)
            
            if current in visited:
                continue
            
            visited.add(current)
            visited_order.append(current)
            
            if current == end:
                path = self.reconstruct_path(parent, end)
                self.execution_time = time.time() - start_time
                self.visited = visited_order
                self.path = path
                return {
                    'found': True,
                    'path': path,
                    'visited': visited_order,
                    'distance': distance[end],
                    'cells_visited': len(visited_order),
                    'time': self.execution_time,
                    'algorithm': 'A*'
                }
            
            for neighbor in self.get_neighbors(current):
                new_dist = distance[current] + 1
                if new_dist < distance[neighbor]:
                    distance[neighbor] = new_dist
                    parent[neighbor] = current
                    f_score = new_dist + heuristic(neighbor)
                    heapq.heappush(pq, (f_score, counter, neighbor))
                    counter += 1
        
        self.execution_time = time.time() - start_time
        self.visited = visited_order
        return {
            'found': False,
            'visited': visited_order,
            'cells_visited': len(visited_order),
            'time': self.execution_time,
            'algorithm': 'A*'
        }

# ============================================================================
# FLASK ROUTES
# ============================================================================

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/api/pathfind', methods=['POST'])
def pathfind():
    """API endpoint for pathfinding"""
    try:
        data = request.json
        grid = data.get('grid')
        start = tuple(data.get('start'))
        end = tuple(data.get('end'))
        algorithm = data.get('algorithm', 'dijkstra')
        
        visualizer = PathfindingVisualizer(grid)
        
        if algorithm == 'dijkstra':
            result = visualizer.dijkstra(start, end)
        elif algorithm == 'bfs':
            result = visualizer.bfs(start, end)
        elif algorithm == 'dfs':
            result = visualizer.dfs(start, end)
        elif algorithm == 'astar':
            result = visualizer.a_star(start, end)
        else:
            result = visualizer.dijkstra(start, end)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/generate-maze', methods=['POST'])
def generate_maze():
    """Generate a random maze"""
    try:
        import random
        data = request.json
        rows = data.get('rows', 20)
        cols = data.get('cols', 20)
        wall_probability = data.get('wall_probability', 0.3)
        
        grid = [[0 if random.random() > wall_probability else 1 
                for _ in range(cols)] for _ in range(rows)]
        
        # Ensure start and end are not walls
        grid[0][0] = 0
        grid[rows-1][cols-1] = 0
        
        return jsonify({'grid': grid})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/maze-templates', methods=['GET'])
def maze_templates():
    """Get predefined maze templates"""
    templates = {
        'simple': generate_simple_maze(),
        'complex': generate_complex_maze(),
        'sparse': generate_sparse_maze()
    }
    return jsonify(templates)

def generate_simple_maze():
    """Generate a simple maze template"""
    maze = [[0]*20 for _ in range(20)]
    for i in range(5, 15):
        for j in range(0, 8):
            if j != 3:
                maze[i][j] = 1
    for i in range(10, 20):
        for j in range(12, 20):
            if i != 15:
                maze[i][j] = 1
    return maze

def generate_complex_maze():
    """Generate a complex maze template"""
    maze = [[0]*20 for _ in range(20)]
    # Add vertical walls
    for i in range(20):
        if i % 4 != 0:
            for j in range(0, 20, 4):
                maze[i][j] = 1
    # Add horizontal walls
    for j in range(20):
        if j % 4 != 0:
            for i in range(0, 20, 4):
                maze[i][j] = 1
    maze[0][0] = 0
    maze[19][19] = 0
    return maze

def generate_sparse_maze():
    """Generate a sparse maze template"""
    maze = [[0]*20 for _ in range(20)]
    for i in range(0, 20, 5):
        for j in range(0, 20, 5):
            maze[i][j] = 1
    maze[0][0] = 0
    maze[19][19] = 0
    return maze

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get algorithm statistics"""
    stats = {
        'algorithms': [
            {
                'name': 'Dijkstra',
                'time_complexity': 'O((V+E) log V)',
                'space_complexity': 'O(V)',
                'guarantee': 'Shortest Path',
                'description': 'Greedy algorithm for weighted graphs'
            },
            {
                'name': 'BFS',
                'time_complexity': 'O(V+E)',
                'space_complexity': 'O(V)',
                'guarantee': 'Shortest Path (unweighted)',
                'description': 'Explores level by level'
            },
            {
                'name': 'DFS',
                'time_complexity': 'O(V+E)',
                'space_complexity': 'O(V)',
                'guarantee': 'Any Path',
                'description': 'Explores deeply first'
            },
            {
                'name': 'A*',
                'time_complexity': 'O((V+E) log V)',
                'space_complexity': 'O(V)',
                'guarantee': 'Shortest Path (with heuristic)',
                'description': 'Uses Manhattan distance heuristic'
            }
        ]
    }
    return jsonify(stats)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
