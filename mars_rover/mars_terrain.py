import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch


class Terrain:
    def __init__(self, n=2, m=4, home_base=(0, 0), sandstorm_time=30):
        self.n = n
        self.m = m
        self.n_waypoints = n * m
        self.home_base = home_base
        self.sandstorm_time = sandstorm_time
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
        self.time_left = self.sandstorm_time
        self.visited = set()
        self.discovered = np.zeros(self.n_waypoints, dtype=np.float32)

        cards_grid = [None] * self.n_waypoints

        # 1. Place "life_detected" in a random corner
        corners = [self.n - 1, self.n * (self.m - 1), self.n * self.m - 1]
        life_idx = np.random.choice(corners)
        cards_grid[life_idx] = self.card_labels["life_detected"]
        life_x, life_y = life_idx % self.n, life_idx // self.n

        # 2. Place "take_soil_sample" adjacent to "life_detected"
        adjacents = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = life_x + dx, life_y + dy
            if 0 <= nx < self.n and 0 <= ny < self.m:
                adjacents.append(ny * self.n + nx)
        if adjacents:
            soil_idx = np.random.choice(adjacents)
            cards_grid[soil_idx] = self.card_labels["take_soil_sample"]
        else:
            # fallback: random empty
            soil_idx = np.random.choice(
                [i for i, c in enumerate(cards_grid) if c is None]
            )
            cards_grid[soil_idx] = self.card_labels["take_soil_sample"]

        # 3. Place "call_sample_retrieval_lander" in same row as "life_detected", not adjacent
        row_indices = [
            life_y * self.n + x
            for x in range(self.n)
            if (life_y * self.n + x) != life_idx and (life_y * self.n + x) != soil_idx
        ]
        if row_indices:
            lander_idx = np.random.choice(row_indices)
            cards_grid[lander_idx] = self.card_labels["call_sample_retrieval_lander"]
        else:
            lander_idx = np.random.choice(
                [i for i, c in enumerate(cards_grid) if c is None]
            )
            cards_grid[lander_idx] = self.card_labels["call_sample_retrieval_lander"]

        # 4. Place "alert" not in same row or column as "life_detected"
        alert_candidates = [
            i
            for i in range(self.n_waypoints)
            if (i // self.n != life_y)
            and (i % self.n != life_x)
            and cards_grid[i] is None
        ]
        if alert_candidates:
            alert_idx = np.random.choice(alert_candidates)
            cards_grid[alert_idx] = self.card_labels["alert"]
        else:
            alert_idx = np.random.choice(
                [i for i, c in enumerate(cards_grid) if c is None]
            )
            cards_grid[alert_idx] = self.card_labels["alert"]

        # 5. Fill remaining with other cards randomly
        used = {
            self.card_labels["life_detected"],
            self.card_labels["take_soil_sample"],
            self.card_labels["call_sample_retrieval_lander"],
            self.card_labels["alert"],
        }
        remaining_cards = [v for k, v in self.card_labels.items() if v not in used]
        np.random.shuffle(remaining_cards)
        for idx, card in zip(
            [i for i, c in enumerate(cards_grid) if c is None], remaining_cards
        ):
            cards_grid[idx] = card

        self.cards_grid = np.array(cards_grid)
        return self.get_state()

    def move_to_waypoint(self, action):
        x = action % self.n
        y = action // self.n
        self.rover_pos = [x, y]
        self.execute_rover_controls(self.rover_pos)
        self.time_left -= 1

    def execute_rover_controls(self, pos):
        pass

    def get_current_card(self):
        idx = self.rover_pos[1] * self.n + self.rover_pos[0]
        return self.cards_grid[idx], idx

    def get_reward(self):
        card_type, _ = self.get_current_card()
        reward = self.reward_table.get(card_type, 0)
        # Penalize step
        reward -= 5
        # Penalize revisiting waypoints (except the first visit)
        if tuple(self.rover_pos) in self.visited and len(self.visited) > 1:
            reward -= 20  # You can adjust the penalty value
        if tuple(self.rover_pos) == self.home_base and len(self.visited) > 0:
            reward += 50
        if self.time_left <= 0:
            reward -= 100
        return reward

    def step(self, action):
        self.move_to_waypoint(action)
        card_type, idx = self.get_current_card()
        self.discovered[idx] = 1.0  # Mark as discovered
        reward = self.get_reward()
        self.visited.add(tuple(self.rover_pos))
        done = (self.time_left <= 0) or (
            tuple(self.rover_pos) == self.home_base and len(self.visited) > 1
        )
        obs = self.get_state()
        return obs, reward, done, {}

    def get_state(self):
        # State: rover position (2), current card (1), time left (1), discovered (8)
        card_type, _ = self.get_current_card()
        return np.array(
            self.rover_pos + [card_type] + [self.time_left] + self.discovered.tolist(),
            dtype=np.float32,
        )

    def get_state_tensor(self, device):
        return torch.FloatTensor(self.get_state()).unsqueeze(0).to(device)

    def plot_grid(
        self, show_rover=True, discovered=None, ax=None, title=None, path=None
    ):
        card_names = {v: k.replace("_", "\n") for k, v in self.card_labels.items()}
        colors = {
            "take_soil_sample": "#b3e2cd",
            "alert": "#fdcdac",
            "call_sample_retrieval_lander": "#cbd5e8",
            "rendezvous_point": "#f4cae4",
            "headlight_switch": "#e6f5c9",
            "cam_neck_prospect": "#fff2ae",
            "radio_signal_to_ingenuity": "#f1e2cc",
            "life_detected": "#fbb4ae",
        }
        if ax is None:
            fig, ax = plt.subplots(figsize=(self.n * 2, self.m * 2))
        ax.set_xlim(-0.5, self.n - 0.5)
        ax.set_ylim(-0.5, self.m - 0.5)
        ax.set_xticks(range(self.n))
        ax.set_yticks(range(self.m))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True, which="both", color="gray", linewidth=1, alpha=0.5)
        ax.set_aspect("equal")
        if title:
            ax.set_title(title, fontsize=14)

        # Draw cards and highlight home base
        for y in range(self.m):
            for x in range(self.n):
                idx = y * self.n + x
                card_val = self.cards_grid[idx]
                card_name = card_names[card_val]
                color = colors.get(card_name.replace("\n", "_"), "#dddddd")
                rect = patches.Rectangle(
                    (x - 0.5, y - 0.5),
                    1,
                    1,
                    linewidth=2,
                    edgecolor="black",
                    facecolor=color,
                    alpha=0.8,
                )
                ax.add_patch(rect)
                # Highlight home base
                if (x, y) == tuple(self.home_base):
                    home_rect = patches.Rectangle(
                        (x - 0.5, y - 0.5),
                        1,
                        1,
                        linewidth=3,
                        edgecolor="green",
                        facecolor="none",
                        zorder=3,
                    )
                    ax.add_patch(home_rect)
                    ax.text(
                        x,
                        y + 0.3,
                        "HOME",
                        ha="center",
                        va="center",
                        fontsize=12,
                        color="green",
                        weight="bold",
                        zorder=4,
                    )
                # Show discovered status if provided
                if discovered is not None and discovered[idx] < 0.5:
                    ax.add_patch(
                        patches.Rectangle(
                            (x - 0.5, y - 0.5), 1, 1, color="black", alpha=0.3
                        )
                    )
                # Card label
                ax.text(
                    x,
                    y,
                    card_name,
                    ha="center",
                    va="center",
                    fontsize=10,
                    weight="bold",
                )
        # Draw rover path
        if path is not None and len(path) > 1:
            xs, ys = zip(*path)
            ax.plot(
                xs,
                ys,
                color="blue",
                linewidth=2,
                marker=".",
                markersize=10,
                alpha=0.7,
                zorder=2,
            )
        # Draw rover
        if show_rover:
            rx, ry = self.rover_pos
            ax.plot(
                rx,
                ry,
                marker="o",
                markersize=30,
                markeredgecolor="red",
                markerfacecolor="none",
                markeredgewidth=3,
            )
            ax.text(
                rx,
                ry,
                "ROVER",
                ha="center",
                va="center",
                fontsize=8,
                color="red",
                weight="bold",
            )

        plt.tight_layout()
        if ax is None:
            plt.show()
