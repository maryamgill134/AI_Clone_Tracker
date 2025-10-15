# import cv2
# import mediapipe as mp
# import numpy as np
# import matplotlib.pyplot as plt
# import time
# import math
# import random

# # Initialize Mediapipe Pose
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose(static_image_mode=False, model_complexity=0, smooth_landmarks=True,
#                      min_detection_confidence=0.5, min_tracking_confidence=0.5)

# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# POSE_CONNECTIONS = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
#                     (11, 23), (12, 24), (23, 24), (23, 25), (24, 26),
#                     (25, 27), (26, 28), (27, 31), (28, 32)]

# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')
# ax.set_facecolor('black')
# plt.ion()

# start_time = time.time()
# frame_count = 0
# trail_points = []
# max_trail_length = 8
# particles = []  # for motion sparks

# # -------------------------------------------
# # Helper Functions
# # -------------------------------------------
# def get_gradient_color(t):
#     hue = (t * 120) % 360
#     c = 1
#     x = c * (1 - abs((hue / 60) % 2 - 1))
#     if 0 <= hue < 60:
#         r, g, b = c, x, 0
#     elif 60 <= hue < 120:
#         r, g, b = x, c, 0
#     elif 120 <= hue < 180:
#         r, g, b = 0, c, x
#     elif 180 <= hue < 240:
#         r, g, b = 0, x, c
#     elif 240 <= hue < 300:
#         r, g, b = x, 0, c
#     else:
#         r, g, b = c, 0, x
#     return (int(r * 255), int(g * 255), int(b * 255))

# def draw_smooth_line(img, pt1, pt2, color, thickness):
#     cv2.line(img, pt1, pt2, color, thickness)
#     cv2.line(img, pt1, pt2, tuple(c//2 for c in color), thickness+2)

# def draw_smooth_circle(img, center, radius, color):
#     cv2.circle(img, center, radius, color, -1)
#     cv2.circle(img, center, radius//2, (255, 255, 255), -1)

# # -------------------------------------------
# # Main Loop
# # -------------------------------------------
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame = cv2.flip(frame, 1)
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = pose.process(frame_rgb)

#     current_time = time.time()
#     wave_time = current_time - start_time

#     gradient_color = get_gradient_color(wave_time)
#     pulse_intensity = 0.8 + 0.2 * math.sin(wave_time * 6)
#     neon_color = tuple(int(c * pulse_intensity) for c in gradient_color)

#     height, width, _ = frame.shape
#     glow_layer = np.zeros_like(frame, dtype=np.uint8)

#     if results.pose_landmarks:
#         # Landmarks
#         clone_landmarks_2d = [(int((1 - lm.x) * width), int(lm.y * height)) for lm in results.pose_landmarks.landmark]
#         clone_landmarks_3d = [(1 - lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]

#         # ---------------------
#         # 💫 Mirror Clone Effect (Dual Colors)
#         # ---------------------
#         alt_color = get_gradient_color(wave_time + 1.5)
#         for i, j in POSE_CONNECTIONS:
#             x1, y1 = clone_landmarks_2d[i]
#             x2, y2 = clone_landmarks_2d[j]
#             draw_smooth_line(glow_layer, (x1, y1), (x2, y2), neon_color, 6)
#             draw_smooth_line(glow_layer, (width - x1, y1), (width - x2, y2), alt_color, 6)

#         # ---------------------
#         # 💥 Joint Points + Energy Pulse Ring
#         # ---------------------
#         clone_joint_points = []
#         for (x, y) in clone_landmarks_2d:
#             if 0 <= x < width and 0 <= y < height:
#                 clone_joint_points.append((x, y))
#                 draw_smooth_circle(glow_layer, (x, y), 8, neon_color)
#                 # Energy ring
#                 ring_radius = int(15 + 5 * abs(math.sin(wave_time * 3)))
#                 cv2.circle(glow_layer, (x, y), ring_radius, (255, 255, 255), 1)

#         # ---------------------
#         # ⚡ Motion Trail Particles (Sparks)
#         # ---------------------
#         for (x, y) in random.sample(clone_joint_points, k=min(6, len(clone_joint_points))):
#             particles.append([x, y, random.uniform(-1, 1), random.uniform(-2, 0), random.randint(10, 20)])
#         for p in particles[:]:
#             x, y, vx, vy, life = p
#             if life <= 0:
#                 particles.remove(p)
#                 continue
#             x += vx
#             y += vy
#             life -= 1
#             cv2.circle(glow_layer, (int(x), int(y)), 2, neon_color, -1)
#             p[0], p[1], p[4] = x, y, life

#         # ---------------------
#         # 🌀 Body Trail (Memory Shadow)
#         # ---------------------
#         trail_points.append(clone_joint_points.copy())
#         if len(trail_points) > max_trail_length:
#             trail_points.pop(0)
#         for t_idx, trail_frame in enumerate(trail_points[:-1]):
#             trail_alpha = (t_idx / len(trail_points)) * 0.4
#             trail_color = tuple(int(c * trail_alpha) for c in neon_color)
#             for x, y in trail_frame:
#                 cv2.circle(glow_layer, (x, y), 3, trail_color, -1)

#         # ---------------------
#         # ⚙️ Energy Charge Bar (Activity Meter)
#         # ---------------------
#         energy = min(1.0, abs(math.sin(wave_time * 2)))
#         cv2.rectangle(frame, (50, height - 40), (int(50 + energy * (width - 100)), height - 20), neon_color, -1)
#         cv2.putText(frame, "ENERGY", (50, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, neon_color, 2)

#         # Blend
#         frame = cv2.addWeighted(frame, 0.6, glow_layer, 0.8, 0)

#         # ---------------------
#         # 🌌 3D Auto Rotating Skeleton
#         # ---------------------
#         if frame_count % 6 == 0:
#             ax.clear()
#             ax.set_xlim([-1, 1])
#             ax.set_ylim([-1, 1])
#             ax.set_zlim([-1, 1])
#             ax.view_init(20, (wave_time * 40) % 360)
#             ax.set_facecolor('black')
#             ax.set_title("3D Holographic Clone", color='cyan', fontsize=12)
#             ax.grid(False)
#             ax.set_xticks([])
#             ax.set_yticks([])
#             ax.set_zticks([])
#             line_color = tuple(c/255 for c in neon_color)
#             for i, j in POSE_CONNECTIONS:
#                 x_vals, y_vals, z_vals = zip(clone_landmarks_3d[i], clone_landmarks_3d[j])
#                 ax.plot(x_vals, y_vals, z_vals, color=line_color, linewidth=3, alpha=0.8)
#             for x, y, z in clone_landmarks_3d:
#                 ax.scatter(x, y, z, c=[line_color], s=50, alpha=0.8)
#             plt.draw()
#             plt.pause(0.001)

#     else:
#         # Searching animation
#         search_text = "SCANNING..."
#         font_scale = 1.0
#         thickness = 2
#         text_size = cv2.getTextSize(search_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
#         text_x = (width - text_size[0]) // 2
#         text_y = (height + text_size[1]) // 2
#         scan_color = get_gradient_color(wave_time * 2)
#         cv2.putText(frame, search_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
#                    font_scale, scan_color, thickness)

#     # FPS
#     if frame_count % 30 == 0:
#         fps = int(1.0 / (time.time() - current_time + 0.001))
#         cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#     cv2.imshow("CLONE TRACKER EVOLUTION", frame)
#     frame_count += 1

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
# plt.ioff()
# plt.close()

import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import random

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=0, smooth_landmarks=True,
                     min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

POSE_CONNECTIONS = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
                    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26),
                    (25, 27), (26, 28), (27, 31), (28, 32)]

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('black')
plt.ion()

start_time = time.time()
frame_count = 0
trail_points = []
ghost_frames = []
max_trail_length = 8
particles = []

# -------------------------------------------
# Helper Functions
# -------------------------------------------
def get_gradient_color(t):
    hue = (t * 120) % 360
    c = 1
    x = c * (1 - abs((hue / 60) % 2 - 1))
    if 0 <= hue < 60:
        r, g, b = c, x, 0
    elif 60 <= hue < 120:
        r, g, b = x, c, 0
    elif 120 <= hue < 180:
        r, g, b = 0, c, x
    elif 180 <= hue < 240:
        r, g, b = 0, x, c
    elif 240 <= hue < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x
    return (int(r * 255), int(g * 255), int(b * 255))

def draw_smooth_line(img, pt1, pt2, color, thickness):
    cv2.line(img, pt1, pt2, color, thickness)
    cv2.line(img, pt1, pt2, tuple(c//2 for c in color), thickness+2)

def draw_smooth_circle(img, center, radius, color):
    cv2.circle(img, center, radius, color, -1)
    cv2.circle(img, center, radius//2, (255, 255, 255), -1)

def draw_hologram_grid(img, t):
    h, w, _ = img.shape
    grid = np.zeros_like(img, dtype=np.uint8)
    step = 40
    for x in range(0, w, step):
        y_shift = int(10 * math.sin(t * 2 + x * 0.05))
        cv2.line(grid, (x, 0), (x, h), (0, 255, 255), 1)
        cv2.line(grid, (0, (x + y_shift) % h), (w, (x + y_shift) % h), (0, 255, 255), 1)
    return cv2.addWeighted(img, 1.0, grid, 0.25, 0)

# -------------------------------------------
# Main Loop
# -------------------------------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    current_time = time.time()
    wave_time = current_time - start_time

    gradient_color = get_gradient_color(wave_time)
    pulse_intensity = 0.8 + 0.2 * math.sin(wave_time * 6)
    neon_color = tuple(int(c * pulse_intensity) for c in gradient_color)

    height, width, _ = frame.shape
    glow_layer = np.zeros_like(frame, dtype=np.uint8)

    # 🔹 Add hologram background grid
    frame = draw_hologram_grid(frame, wave_time)

    if results.pose_landmarks:
        clone_landmarks_2d = [(int((1 - lm.x) * width), int(lm.y * height)) for lm in results.pose_landmarks.landmark]
        clone_landmarks_3d = [(1 - lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]

        # ---------------------
        # Ghost Projection Layer
        # ---------------------
        ghost_layer = np.zeros_like(frame, dtype=np.uint8)
        for i, j in POSE_CONNECTIONS:
            x1, y1 = clone_landmarks_2d[i]
            x2, y2 = clone_landmarks_2d[j]
            draw_smooth_line(ghost_layer, (x1, y1), (x2, y2), (150, 50, 255), 3)
        ghost_frames.append(ghost_layer)
        if len(ghost_frames) > 10:
            ghost_frames.pop(0)
        for idx, g in enumerate(ghost_frames):
            alpha = idx / len(ghost_frames) * 0.3
            frame = cv2.addWeighted(frame, 1, g, alpha, 0)

        # ---------------------
        # Clone Body & Trail
        # ---------------------
        for i, j in POSE_CONNECTIONS:
            x1, y1 = clone_landmarks_2d[i]
            x2, y2 = clone_landmarks_2d[j]
            draw_smooth_line(glow_layer, (x1, y1), (x2, y2), neon_color, 6)

        clone_joint_points = []
        for (x, y) in clone_landmarks_2d:
            if 0 <= x < width and 0 <= y < height:
                clone_joint_points.append((x, y))
                draw_smooth_circle(glow_layer, (x, y), 8, neon_color)
                ring_radius = int(15 + 5 * abs(math.sin(wave_time * 3)))
                cv2.circle(glow_layer, (x, y), ring_radius, (255, 255, 255), 1)

        # ---------------------
        # Orbit Drones (around head & hands)
        # ---------------------
        important_joints = [0, 15, 16]  # nose, left_hand, right_hand
        for j_idx in important_joints:
            x, y = clone_landmarks_2d[j_idx]
            orbit_r = 20 + 10 * math.sin(wave_time * 3)
            for angle in range(0, 360, 120):
                ox = int(x + orbit_r * math.cos(math.radians(angle + wave_time * 120)))
                oy = int(y + orbit_r * math.sin(math.radians(angle + wave_time * 120)))
                cv2.circle(glow_layer, (ox, oy), 6, get_gradient_color(wave_time + angle / 100), -1)

        # ---------------------
        # Particle Sparks
        # ---------------------
        for (x, y) in random.sample(clone_joint_points, k=min(5, len(clone_joint_points))):
            particles.append([x, y, random.uniform(-1, 1), random.uniform(-2, 0), random.randint(10, 20)])
        for p in particles[:]:
            x, y, vx, vy, life = p
            if life <= 0:
                particles.remove(p)
                continue
            x += vx
            y += vy
            life -= 1
            cv2.circle(glow_layer, (int(x), int(y)), 2, neon_color, -1)
            p[0], p[1], p[4] = x, y, life

        # ---------------------
        # Energy Bar
        # ---------------------
        energy = min(1.0, abs(math.sin(wave_time * 2)))
        cv2.rectangle(frame, (50, height - 40), (int(50 + energy * (width - 100)), height - 20), neon_color, -1)
        cv2.putText(frame, "ENERGY", (50, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, neon_color, 2)

        # Merge layers
        frame = cv2.addWeighted(frame, 0.6, glow_layer, 0.8, 0)

        # ---------------------
        # Rotating 3D Pose View
        # ---------------------
        if frame_count % 6 == 0:
            ax.clear()
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_zlim([-1, 1])
            ax.view_init(20, (wave_time * 40) % 360)
            ax.set_facecolor('black')
            ax.set_title("3D Quantum Skeleton", color='cyan', fontsize=12)
            ax.grid(False)
            line_color = tuple(c/255 for c in neon_color)
            for i, j in POSE_CONNECTIONS:
                x_vals, y_vals, z_vals = zip(clone_landmarks_3d[i], clone_landmarks_3d[j])
                ax.plot(x_vals, y_vals, z_vals, color=line_color, linewidth=3, alpha=0.8)
            for x, y, z in clone_landmarks_3d:
                ax.scatter(x, y, z, c=[line_color], s=50, alpha=0.8)
            plt.draw()
            plt.pause(0.001)

    else:
        # Scanning animation
        search_text = "INITIALIZING AI CLONE..."
        font_scale = 1.0
        thickness = 2
        text_size = cv2.getTextSize(search_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        text_x = (width - text_size[0]) // 2
        text_y = (height + text_size[1]) // 2
        scan_color = get_gradient_color(wave_time * 2)
        cv2.putText(frame, search_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale, scan_color, thickness)

    # FPS
    if frame_count % 30 == 0:
        fps = int(1.0 / (time.time() - current_time + 0.001))
        cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("CLONE TRACKER - HOLOGRAM MODE", frame)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.close()
