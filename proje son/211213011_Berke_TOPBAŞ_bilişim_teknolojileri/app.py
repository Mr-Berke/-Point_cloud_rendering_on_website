import os
import sqlite3
from flask import Flask, render_template, request, jsonify
import threading
import open3d as o3d
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

conn = sqlite3.connect('processed_files.db', check_same_thread=False)
cursor = conn.cursor()

cursor.execute('''CREATE TABLE IF NOT EXISTS files (
                    id INTEGER PRIMARY KEY,
                    file_name TEXT,
                    upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    file_path TEXT)''')

cursor.execute('''CREATE TABLE IF NOT EXISTS points (
                    id INTEGER PRIMARY KEY,
                    file_id INTEGER,
                    x REAL,
                    y REAL,
                    z REAL,
                    FOREIGN KEY (file_id) REFERENCES files(id))''')

cursor.execute('''CREATE TABLE IF NOT EXISTS polygons (
                    id INTEGER PRIMARY KEY,
                    file_id INTEGER,
                    corner_num INTEGER,
                    normal_x REAL,
                    normal_y REAL,
                    normal_z REAL,
                    FOREIGN KEY (file_id) REFERENCES files(id))''')

cursor.execute('''CREATE TABLE IF NOT EXISTS normals (
                    id INTEGER PRIMARY KEY,
                    file_id INTEGER,
                    polygon_id INTEGER,
                    normal_x REAL,
                    normal_y REAL,
                    normal_z REAL,
                    FOREIGN KEY (file_id) REFERENCES files(id),
                    FOREIGN KEY (polygon_id) REFERENCES polygons(id))''')

conn.commit()

class PLYProcessor:
    def __init__(self):
        self.file_path = None
        self.file_id = None

    def set_file_path(self, file_path):
        self.file_path = file_path

    def save_file_to_db(self, file_name, file_path):
        cursor.execute('INSERT INTO files (file_name, file_path) VALUES (?, ?)', (file_name, file_path))
        conn.commit()
        self.file_id = cursor.lastrowid

    def save_point_to_db(self, x, y, z):
        cursor.execute('INSERT INTO points (file_id, x, y, z) VALUES (?, ?, ?, ?)', (self.file_id, x, y, z))
        conn.commit()

    def save_polygon_to_db(self, corner_num, normal_x, normal_y, normal_z):
        cursor.execute('INSERT INTO polygons (file_id, corner_num, normal_x, normal_y, normal_z) VALUES (?, ?, ?, ?, ?)', 
                       (self.file_id, corner_num, normal_x, normal_y, normal_z))
        conn.commit()
        return cursor.lastrowid

    def save_normal_to_db(self, polygon_id, normal_x, normal_y, normal_z):
        cursor.execute('INSERT INTO normals (file_id, polygon_id, normal_x, normal_y, normal_z) VALUES (?, ?, ?, ?, ?)', 
                       (self.file_id, polygon_id, normal_x, normal_y, normal_z))
        conn.commit()

    def open_file(self):
        if self.file_path:
            point_cloud = o3d.io.read_point_cloud(self.file_path)
            for point in np.asarray(point_cloud.points):
              self.save_point_to_db(*point)
            o3d.visualization.draw_geometries([point_cloud])

    def find_planes(self):
        if self.file_path:
            point_cloud = o3d.io.read_point_cloud(self.file_path)
            plane_model, inliers = point_cloud.segment_plane(distance_threshold=0.05, ransac_n=5, num_iterations=100000)
            inlier_cloud = point_cloud.select_by_index(inliers)
           # for point in np.asarray(inlier_cloud.points):
               # self.save_point_to_db(*point)
            o3d.visualization.draw_geometries([inlier_cloud])

    def find_multiple_planes(self):
        if self.file_path:
            point_cloud = o3d.io.read_point_cloud(self.file_path)
            num_planes = 4
            inlier_clouds = []
            for _ in range(num_planes):
                plane_model, inliers = point_cloud.segment_plane(distance_threshold=0.05, ransac_n=5, num_iterations=100000)
                inlier_cloud = point_cloud.select_by_index(inliers)
                inlier_cloud.paint_uniform_color([np.random.rand(), np.random.rand(), np.random.rand()])
               # for point in np.asarray(inlier_cloud.points):
                   # self.save_point_to_db(*point)
                inlier_clouds.append(inlier_cloud)
                point_cloud = point_cloud.select_by_index(inliers, invert=True)
            o3d.visualization.draw_geometries(inlier_clouds)

    def find_points_above_threshold(self):
        if self.file_path:
            point_cloud = o3d.io.read_point_cloud(self.file_path)
            planes, result = self.find_planes_above_threshold(point_cloud, min_points_per_plane=500)
            o3d.visualization.draw_geometries(result)

    def find_convex_hull_with_normals(self):
        if self.file_path:
            point_cloud = o3d.io.read_point_cloud(self.file_path)
            planes, hulls, normal_lines = self.find_planes_and_hulls(point_cloud, min_points_per_plane=500)
            o3d.visualization.draw_geometries(hulls + normal_lines)

    def find_planes_above_threshold(self, point_cloud, min_points_per_plane=500):
        planes = []
        inlier_clouds = []
        while len(point_cloud.points) >= min_points_per_plane:
            plane_model, inliers = point_cloud.segment_plane(distance_threshold=0.05, ransac_n=10, num_iterations=100000)
            inlier_cloud = point_cloud.select_by_index(inliers)
            inlier_cloud.paint_uniform_color([np.random.rand(), np.random.rand(), np.random.rand()])
          #  for point in np.asarray(inlier_cloud.points):
               # self.save_point_to_db(*point)
            inlier_clouds.append(inlier_cloud)
            point_cloud = point_cloud.select_by_index(inliers, invert=True)
            if len(inliers) < min_points_per_plane:
                break
            planes.append(plane_model)
        return planes, inlier_clouds

    def find_planes_and_hulls(self, point_cloud, min_points_per_plane=500, distance_threshold=0.05, ransac_n=10, num_iterations=100000):
        planes = []
        hulls = []
        normal_lines = []
        while len(point_cloud.points) >= min_points_per_plane:
            plane_model, inliers = point_cloud.segment_plane(distance_threshold=distance_threshold, ransac_n=ransac_n, num_iterations=num_iterations)
            inlier_cloud = point_cloud.select_by_index(inliers)
          #  for point in np.asarray(inlier_cloud.points):
                #self.save_point_to_db(*point)
            hull, _ = inlier_cloud.compute_convex_hull()
            hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
            hull_ls.paint_uniform_color([np.random.rand(), np.random.rand(), np.random.rand()])

            vertices = np.asarray(hull.vertices)
            triangles = np.asarray(hull.triangles)
            polygon_id = self.save_polygon_to_db(len(triangles), *plane_model[:3])

            for triangle in triangles:
                v0 = vertices[triangle[0]]
                v1 = vertices[triangle[1]]
                v2 = vertices[triangle[2]]
                center = (v0 + v1 + v2) / 3
                normal = np.cross(v1 - v0, v2 - v0)
                normal = normal / np.linalg.norm(normal)

                self.save_normal_to_db(polygon_id, *normal)

                line_set = o3d.geometry.LineSet(
                    points=o3d.utility.Vector3dVector([center, center + normal * 0.1]),
                    lines=o3d.utility.Vector2iVector([[0, 1]]),
                )
                line_set.paint_uniform_color([1, 0, 0])
                normal_lines.append(line_set)

            hulls.append(hull_ls)
            point_cloud = point_cloud.select_by_index(inliers, invert=True)

            if len(inliers) < min_points_per_plane:
                break

            planes.append(plane_model)
            
        return planes, hulls, normal_lines

ply_processor = PLYProcessor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        ply_processor.set_file_path(file_path)
        ply_processor.save_file_to_db(file.filename, file_path)
        return jsonify({'message': 'File uploaded successfully', 'file_path': file_path})

@app.route('/process', methods=['POST'])
def process_file():
    action = request.json.get('action')
    if action == 'open':
        threading.Thread(target=ply_processor.open_file).start()
    elif action == 'find_planes':
        threading.Thread(target=ply_processor.find_planes).start()
    elif action == 'find_multiple_planes':
        threading.Thread(target=ply_processor.find_multiple_planes).start()
    elif action == 'find_points_above_threshold':
        threading.Thread(target=ply_processor.find_points_above_threshold).start()
    elif action == 'find_convex_hull_with_normals':
        threading.Thread(target=ply_processor.find_convex_hull_with_normals).start()
    else:
        return jsonify({'message': 'Invalid action'})
    return jsonify({'message': 'Processing started'})

if __name__ == '__main__':
    app.run(debug=True)
