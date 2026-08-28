import faulthandler, sys
faulthandler.enable()
import gymnasium as gym
import mani_skill.envs

try:
    env = gym.make("PushCube-v1", obs_mode="none", control_mode="pd_joint_pos", render_mode="rgb_array")
    print("ENV CREATED OK")
except Exception as e:
    import traceback
    traceback.print_exc()
    print("CAUGHT PYTHON EXCEPTION:", e)
    sys.exit(0)
