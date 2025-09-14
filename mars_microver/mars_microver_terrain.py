import math

import matplotlib.pyplot as plt
import numpy as np
import torch


class Terrain:
    def __init__(self, width=4.0, height=7.0, sandstorm_time=30):
        self.width = width
        self.height = height
        self.home_base = (width / 2, 0.0)
        self.sandstorm_time = sandstorm_time
        self.step_size = 0.2  # feet
        self.noise_std = 0.05  # movement noise
        self.directions = {
            0: (0, 1),
            1: (1, 1),
            2: (1, 0),
            3: (1, -1),
            4: (0, -1),
            5: (-1, -1),
            6: (-1, 0),
            7: (-1, 1),
        }
        self.card_labels = {
            "take_soil_sample": 0,
            "alert": 1,
            "call_sample_retrieval_lander": 2,
            "rendezvous_point": 3,
            "headlight_switch": 4,
            "cam_neck_prospect": 5,
            "radio_signal_to_ingenuity": 6,
            "life_detected": 7,
        }
        self.reward_table = {0: 15, 1: -5, 2: 10, 3: 3, 4: 2, 5: 6, 6: 4, 7: 25}
        self.reset()

    def reset(self):
        self.rover_pos = list(self.home_base)
        self.camera_angle = 90.0
        self.time_left = self.sandstorm_time
        self.scanned_cards = set()
        self.cards = []
        self.path = []
        self.camera_angles = [self.camera_angle]
        self.has_left_home = False
        self.home_reward_given = False
        self.rover_heading = 90.0  # Facing "up" by default
        self.min_dist_home_after_life = None

        # 1. Place life_detected in upper half
        life_x = np.random.uniform(0.5, self.width - 0.5)
        life_y = np.random.uniform(self.height / 2, self.height - 0.5)
        self.cards.append({"label": "life_detected", "pos": (life_x, life_y)})

        # 2. Place take_soil_sample nearby (~0.5 ft)
        offset1 = np.random.normal(0, 0.25, size=2)
        soil_x = np.clip(life_x + offset1[0], 0, self.width)
        soil_y = np.clip(life_y + offset1[1], 0, self.height)
        self.cards.append({"label": "take_soil_sample", "pos": (soil_x, soil_y)})

        # 3. Place call_sample_retrieval_lander nearby (~0.75 ft)
        offset2 = np.random.normal(0, 0.35, size=2)
        lander_x = np.clip(life_x + offset2[0], 0, self.width)
        lander_y = np.clip(life_y + offset2[1], 0, self.height)
        self.cards.append(
            {"label": "call_sample_retrieval_lander", "pos": (lander_x, lander_y)}
        )

        # 4. Place remaining cards randomly or clustered
        used_labels = {
            "life_detected",
            "take_soil_sample",
            "call_sample_retrieval_lander",
        }
        cluster_center = (self.width * 0.25, self.height * 0.25)
        for label in self.card_labels:
            if label in used_labels:
                continue
            if np.random.rand() < 0.5:
                x = np.random.normal(cluster_center[0], 0.3)
                y = np.random.normal(cluster_center[1], 0.3)
            else:
                x = np.random.uniform(0, self.width)
                y = np.random.uniform(0, self.height)
            self.cards.append({"label": label, "pos": (x, y)})

        return self.get_state()

    def detect_card(self, start_pos, end_pos, angle):
        detected = False
        for card in self.cards:
            cx, cy = card["pos"]
            dx = cx - end_pos[0]
            dy = cy - end_pos[1]
            distance = math.sqrt(dx**2 + dy**2)
            if 0.333 < distance < 1.0:  # 4–12 inches
                heading = math.degrees(math.atan2(dy, dx))  # Range: -180 to +180
                if abs((heading - angle + 180) % 360 - 180) <= 5:
                    if np.random.rand() < 0.5:
                        self.scanned_cards.add(card["label"])
                        detected = True
                        return detected, heading  # retain angle of detection
        return False, None  # no detection

    def step(self, action):
        prev_pos = self.rover_pos.copy()
        detected = False
        detected_angle = None

        if action in range(8):  # movement
            dx, dy = self.directions[action]
            dx += np.random.normal(0, self.noise_std)
            dy += np.random.normal(0, self.noise_std)
            new_x = self.rover_pos[0] + dx * self.step_size
            new_y = self.rover_pos[1] + dy * self.step_size

            detected, detected_angle = self.detect_card(
                prev_pos, (new_x, new_y), self.camera_angle
            )

            self.rover_pos[0] = new_x
            self.rover_pos[1] = new_y
            heading_rad = math.atan2(dy, dx)
            self.rover_heading = math.degrees(heading_rad)

        elif action == 8:  # pan camera
            for angle in range(-90, 91, 5):
                found, found_angle = self.detect_card(
                    self.rover_pos, self.rover_pos, angle
                )
                if found:
                    detected = True
                    detected_angle = found_angle
                    break

        # Update camera angle
        if detected and detected_angle is not None:
            self.camera_angle = detected_angle
        else:
            self.camera_angle = 0.0

        # Update has_left_home flag
        dist_home = math.sqrt(
            (self.rover_pos[0] - self.home_base[0]) ** 2
            + (self.rover_pos[1] - self.home_base[1]) ** 2
        )
        if dist_home > 1.0:
            self.has_left_home = True

        self.time_left -= 1

        if "life_detected" in self.scanned_cards:
            if self.min_dist_home_after_life is None:
                self.min_dist_home_after_life = dist_home
            else:
                self.min_dist_home_after_life = min(
                    self.min_dist_home_after_life, dist_home
                )

        reward = self.get_reward()

        # Check if episode should end
        done = False
        if self.has_left_home and dist_home <= 1.0:
            done = True
        elif self.time_left <= 0:
            done = True

        self.path.append(tuple(self.rover_pos))
        self.camera_angles.append(self.rover_heading + self.camera_angle)
        return self.get_state(), reward, done, {}

    def get_reward(self):
        reward = -1  # Base step penalty

        # --- Unique card rewards ---
        for card in self.cards:
            label = card["label"]
            if label in self.scanned_cards and not hasattr(card, "rewarded"):
                reward += self.reward_table[self.card_labels[label]]
                card["rewarded"] = True  # Mark as rewarded

        # --- Boundary penalty ---
        if not (
            0 <= self.rover_pos[0] <= self.width
            and 0 <= self.rover_pos[1] <= self.height
        ):
            reward -= 50

        # --- Home return bonus (only once, after leaving) ---
        dist_home = math.sqrt(
            (self.rover_pos[0] - self.home_base[0]) ** 2
            + (self.rover_pos[1] - self.home_base[1]) ** 2
        )
        if self.has_left_home and dist_home <= 1.0 and not self.home_reward_given:
            reward += 50
            self.home_reward_given = True

            # --- Bonus for returning with life_detected ---
            if "life_detected" in self.scanned_cards:
                reward += 100  # Mission success bonus

        # --- Storm penalty if not home ---
        if self.time_left <= 0 and dist_home > 1.0:
            reward -= 100

        # --- Penalty for being far from home after life_detected ---
        if "life_detected" in self.scanned_cards and not self.home_reward_given:
            dist_home = math.sqrt(
                (self.rover_pos[0] - self.home_base[0]) ** 2
                + (self.rover_pos[1] - self.home_base[1]) ** 2
            )
            if (
                self.min_dist_home_after_life is not None
                and dist_home > self.min_dist_home_after_life
            ):
                reward -= (
                    dist_home - self.min_dist_home_after_life
                ) * 10  # Tunable weight

        return reward

    def get_state(self):
        scanned = [
            1 if label in self.scanned_cards else 0 for label in self.card_labels
        ]
        return np.array(
            self.rover_pos + [self.camera_angle, self.time_left] + scanned,
            dtype=np.float32,
        )

    def get_state_tensor(self, device):
        return torch.FloatTensor(self.get_state()).unsqueeze(0).to(device)

    def plot_path(self, episode_num):
        path = np.array(self.path)
        fig, ax = plt.subplots(figsize=(6, 10))
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_title("Rover Path")

        # Plot terrain boundary
        ax.plot(
            [0, self.width, self.width, 0, 0],
            [0, 0, self.height, self.height, 0],
            "k--",
            label="Boundary",
        )

        # Plot rover path
        if len(path) > 0:
            ax.plot(path[:, 0], path[:, 1], "b-", label="Path")
            ax.plot(path[0, 0], path[0, 1], "go", label="Start")
            ax.plot(path[-1, 0], path[-1, 1], "ro", label="End")
        import matplotlib.patches as patches

        # Field of view indicator
        for i in range(len(self.path)):
            x, y = self.path[i]
            angle_deg = self.camera_angles[i]
            fov_width = 30
            fov_length = 1.0
            wedge = patches.Wedge(
                center=(x, y),
                r=fov_length,
                theta1=angle_deg - fov_width / 2,
                theta2=angle_deg + fov_width / 2,
                color="orange",
                alpha=0.1,
            )  # Transparent cone
            ax.add_patch(wedge)

        # Plot cards
        for card in self.cards:
            x, y = card["pos"]
            ax.plot(x, y, "ks")
            ax.text(x + 0.05, y + 0.05, card["label"], fontsize=8)

        ax.legend()
        ax.grid(True)
        plt.savefig(f"Micro_Path_e{episode_num + 1}.png")
        plt.close()
