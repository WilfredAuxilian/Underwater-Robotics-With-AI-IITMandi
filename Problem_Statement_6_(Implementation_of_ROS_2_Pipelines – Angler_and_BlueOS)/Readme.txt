1. Prerequisites

Ensure you have:

* Ubuntu 22.04 or 24.04
* ROS 2 Humble, Iron, or Rolling installed and sourced

Run:

```bash
source /opt/ros/<ros_distro>/setup.bash
```

---

## 2. Angler Installation & Launch

Based on the Angler README: ([GitHub][1])

1. **Create a ROS workspace** (if not already):

   ```bash
   mkdir -p ~/ros2_ws/src
   cd ~/ros2_ws
   ```

2. **Clone the Angler repo**, replacing `<ros_distro>`:

   ```bash
   cd src
   git clone -b <ros_distro> https://github.com/Robotic-Decision-Making-Lab/angler.git
   cd ..
   ```

3. **Import additional repos using `vcs`**:

   ```bash
   vcs import src < src/angler/angler.repos
   ```

4. **Install dependencies via rosdep**:

   ```bash
   rosdep update
   rosdep install -y --from-paths src --ignore-src --rosdistro <ros_distro>
   ```

5. **Build the workspace**:

   ```bash
   colcon build --symlink-install
   source install/setup.bash
   ```

6. **Launch Angler in simulation mode for BlueROV2 Heavy + Reach manipulator**:

   ```bash
   ros2 launch angler_bringup bluerov2_heavy_alpha.launch.py use_sim:=true
   ```

Test the arguments:

```bash
ros2 launch angler_bringup bluerov2_heavy_alpha.launch.py --show-args
```

---

## 3. Blue Installation & Launch

Blue is also maintained by RDML and supports sim-to-real deployment for underwater robotics: ([GitHub][1], [GitHub][2])

1. Ensure inside your workspace:

   ```bash
   cd ~/ros2_ws/src
   ```

2. **Clone the Blue repo**:

   ```bash
   git clone https://github.com/Robotic-Decision-Making-Lab/blue.git
   cd ..
   ```

3. **Import additional packages** (similar pattern if applicable—check for `blue.repos`):

   ```bash
   vcs import src < src/blue/blue.repos
   ```

4. **Install dependencies**:

   ```bash
   rosdep update
   rosdep install -y --from-paths src --ignore-src --rosdistro <ros_distro>
   ```

5. **Rebuild workspace (with Angler and Blue combined)**:

   ```bash
   colcon build --symlink-install
   source install/setup.bash
   ```

6. **Run Blue launch file** (check documentation or readme for available launch files):

   ```bash
   ros2 launch blue <appropriate_launch_file>.launch.py
   ```

---

## ✅ Summary Table

| Step            | Command                                        |
| --------------- | ---------------------------------------------- |
| Clone Angler    | `git clone -b <ros_distro> .../angler.git`     |
| Clone Blue      | `git clone .../blue.git`                       |
| Import repos    | `vcs import src < angler/blue.repos`           |
| Install deps    | `rosdep install ...`                           |
| Build workspace | `colcon build --symlink-install`               |
| Source env      | `source install/setup.bash`                    |
| Launch Angler   | `ros2 launch angler_bringup ... use_sim:=true` |
| Launch Blue     | `ros2 launch blue <launch_file>`               |

---

## ℹ️ Notes & Tips

* Angler supports **Humble, Iron, Rolling** branches; choose the one matching your ROS distro. ([0xdf hacks stuff][3], [GitHub][1])
* Blue provides simulation features (Gazebo + ROS 2 + ArduSub SITL) for real-time mission development. ([GitHub][2])
* Ensure **Gazebo**, **MAVROS**, **ArduSub SITL** are properly installed if simulation fails.
* If any packages fail to compile, use `rosdep install` to resolve missing dependencies and rebuild selectively with `colcon build --packages-select`.
