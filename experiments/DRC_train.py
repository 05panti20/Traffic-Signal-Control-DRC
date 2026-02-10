import os
import sys

import gymnasium as gym
from stable_baselines3.dqn.dqn import DQN


# ローカルリポジトリを優先的に読み込む
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import  traci

import sumo_rl
from sumo_rl import SumoEnvironment
from scripts.randomize_flow_probability import randomize_flow_probability
# import gym


if __name__ == "__main__":
    env = SumoEnvironment(
        net_file="sumo_rl/nets/baseline/baseline3.net.xml",
        route_file="sumo_rl/nets/baseline/baseline4_straight_we.rou.xml",
        out_csv_name="outputs/baseline/dqn",
        single_agent=True,
        use_gui=False,
        num_seconds=5000,
        print_step_progress=True,
    )

    # # sumo_rl インストール元確認
    # print("sumo_rl from:", sumo_rl.__file__)



    #     # 観測ベクトルデバック部分 開始
    # obs = env.reset()
    # step_count = 0

    # while True: 
    #     # Take a random action (replace with model's action if needed)
    #     action = env.action_space.sample()
    #     obs, reward, terminated, truncated, info = env.step(action)

    #     # Increment step count
    #     step_count += 1

    #     # Print observation every 100 steps
    #     if step_count % 100 == 0:
    #         print(f"Step {step_count}: Observation: {obs}")

    #     if terminated or truncated:
    #         break
    #     # 観測ベクトルデバック部分 終了


    model = DQN(
        env=env,
        policy="MlpPolicy",
        # policy_kwargs={
        # "net_arch": [256, 64],  # 層数・ニューロン数を変更
        # },
        learning_rate=0.0001,
        learning_starts=10000,
        train_freq=1,
        target_update_interval=5000,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        exploration_fraction=0.8,
        batch_size=64,
        buffer_size=200000,
        verbose=1,
    )
    # 学習設定（総ステップ数と切り替えポイント）
    total_timesteps = 320000
    # ここで何ステップで net を切り替えるかを指定（例: 50000）
    switch_at1 = 20000
    switch_at2 = 40000
    switch_at3 = 60000
    switch_at4 = 80000
    switch_at5 = 100000
    switch_at6 = 120000
    switch_at7 = 140000
    switch_at8 = 160000
    switch_at9 = 180000
    switch_at10 = 200000
    switch_at11 = 220000
    switch_at12 = 270000

    # まず最初の環境で学習を開始
    model.learn(total_timesteps=switch_at1)

    # 中途で net を切り替える場合の処理
    # assumption: 切り替え後のネットとルートファイルを指定する
    new_net1 = "sumo_rl/nets/baseline/baseline3.net.xml"
    new_route1 = "sumo_rl/nets/baseline/baseline4_straight_sn.rou.xml"

    # 古い環境を安全に閉じる
    try:
        env.close()
    except Exception:
        pass

    # new_env1の開始
    # 新しい環境を作成してモデルに設定（学習状態は継続）
    new_env1 = SumoEnvironment(
        net_file=new_net1,
        route_file=new_route1,
        out_csv_name="outputs/baseline/dqn_after_switch",
        single_agent=True,
        use_gui=False,
        num_seconds=5000,
        print_step_progress=True,
    )

    model.set_env(new_env1)
    # If the new route matches the target, try to force the model's exploration rate to 0.5.
    def _try_set_model_exploration(m, value):
        
        names = [
            "exploration_rate",
            "exploration",
            "eps",
            "exploration_initial_eps",
        ]
        for n in names:
            try:
                if hasattr(m, n):
                    setattr(m, n, value)
                    print(f"Set model.{n} = {value}")
                    return True
            except Exception:
                pass

        # Try nested attributes commonly used by some implementations
        try:
            if hasattr(m, "policy") and hasattr(m.policy, "exploration"):
                try:
                    m.policy.exploration = value
                    print(f"Set model.policy.exploration = {value}")
                    return True
                except Exception:
                    pass
        except Exception:
            pass

        print("Could not set a recognized exploration attribute on the model; no-op")
        return False

    if os.path.basename(new_route1) == "baseline4_straight_sn.rou.xml":
        # Try best-effort approaches to force epsilon to 0.5.
        _try_set_model_exploration(model, 0.5)

        # Stable-Baselines3 DQN uses a linear schedule based on
        # `exploration_initial_eps` and `exploration_final_eps`. To force
        # the instantaneous exploration rate to remain at 0.5, set both
        # endpoints to 0.5 so the schedule returns a constant value.
        changed = False
        if hasattr(model, "exploration_initial_eps"):
            try:
                model.exploration_initial_eps = 0.5
                # print("Set model.exploration_initial_eps = 0.5")
                print(f"Set model.exploration_initial_eps = {model.exploration_initial_eps}")
                print(f'Current exploration rate: {model.exploration_rate}')
                print(f'exploration_final_eps: {model.exploration_final_eps}')
                print(f'exploration_fraction: {model.exploration_fraction}')
                changed = True
            except Exception:
                pass
        # if hasattr(model, "exploration_final_eps"):
        #     try:
        #         model.exploration_final_eps = 0.5
        #         print("Set model.exploration_final_eps = 0.5")
        #         changed = True
        #     except Exception:
        #         pass

        # As a fallback, also try to set common policy-level attributes.
        # try:
        #     if hasattr(model, "policy"):
        #         if hasattr(model.policy, "exploration"):
        #             model.policy.exploration = 0.5
        #             print("Set model.policy.exploration = 0.5")
        #             changed = True
        #         if hasattr(model.policy, "epsilon"):
        #             model.policy.epsilon = 0.5
        #             print("Set model.policy.epsilon = 0.5")
        #             changed = True
        # except Exception:
        #     pass

        if not changed:
            print("Warning: could not set DQN schedule endpoints; exploration may continue to decay according to internal schedule")

    remaining = max(0, switch_at2 - switch_at1)
    if remaining > 0:
        model.learn(total_timesteps=remaining, reset_num_timesteps=False)

    new_net2 = "sumo_rl/nets/baseline/baseline3.net.xml"
    new_route2 = "sumo_rl/nets/baseline/baseline4_left_we.rou.xml"

    try:
        env.close()
    except Exception:
        pass


    # new_env2の開始
    # 新しい環境を作成してモデルに設定（学習状態は継続）
    new_env2 = SumoEnvironment(
        net_file=new_net2,
        route_file=new_route2,
        out_csv_name="outputs/baseline/dqn_after_switch",
        single_agent=True,
        use_gui=False,
        num_seconds=5000,
        print_step_progress=True,
    )

    model.set_env(new_env2)
    # If the new route matches the target, try to force the model's exploration rate to 0.5.
    def _try_set_model_exploration(m, value):
        
        names = [
            "exploration_rate",
            "exploration",
            "eps",
            "exploration_initial_eps",
        ]
        for n in names:
            try:
                if hasattr(m, n):
                    setattr(m, n, value)
                    print(f"Set model.{n} = {value}")
                    return True
            except Exception:
                pass

        # Try nested attributes commonly used by some implementations
        try:
            if hasattr(m, "policy") and hasattr(m.policy, "exploration"):
                try:
                    m.policy.exploration = value
                    print(f"Set model.policy.exploration = {value}")
                    return True
                except Exception:
                    pass
        except Exception:
            pass

        print("Could not set a recognized exploration attribute on the model; no-op")
        return False

    if os.path.basename(new_route2) == "baseline4_left_we.rou.xml":
        # Try best-effort approaches to force epsilon to 0.5.
        _try_set_model_exploration(model, 0.5)

        # Stable-Baselines3 DQN uses a linear schedule based on
        # `exploration_initial_eps` and `exploration_final_eps`. To force
        # the instantaneous exploration rate to remain at 0.5, set both
        # endpoints to 0.5 so the schedule returns a constant value.
        changed = False
        if hasattr(model, "exploration_initial_eps"):
            try:
                model.exploration_initial_eps = 0.5
                # print("Set model.exploration_initial_eps = 0.5")
                print(f"Set model.exploration_initial_eps = {model.exploration_initial_eps}")
                print(f'Current exploration rate: {model.exploration_rate}')
                print(f'exploration_final_eps: {model.exploration_final_eps}')
                print(f'exploration_fraction: {model.exploration_fraction}')
                changed = True
            except Exception:
                pass
        # if hasattr(model, "exploration_final_eps"):
        #     try:
        #         model.exploration_final_eps = 0.5
        #         print("Set model.exploration_final_eps = 0.5")
        #         changed = True
        #     except Exception:
        #         pass

        # As a fallback, also try to set common policy-level attributes.
        # try:
        #     if hasattr(model, "policy"):
        #         if hasattr(model.policy, "exploration"):
        #             model.policy.exploration = 0.5
        #             print("Set model.policy.exploration = 0.5")
        #             changed = True
        #         if hasattr(model.policy, "epsilon"):
        #             model.policy.epsilon = 0.5
        #             print("Set model.policy.epsilon = 0.5")
        #             changed = True
        # except Exception:
        #     pass

        if not changed:
            print("Warning: could not set DQN schedule endpoints; exploration may continue to decay according to internal schedule")

    remaining = max(0, switch_at3 - switch_at2)
    if remaining > 0:
        model.learn(total_timesteps=remaining, reset_num_timesteps=False)

    new_net3 = "sumo_rl/nets/baseline/baseline3.net.xml"
    new_route3 = "sumo_rl/nets/baseline/baseline4_left_sn.rou.xml"

    try:
        env.close()
    except Exception:
        pass

    # new_env3の開始
    # 新しい環境を作成してモデルに設定（学習状態は継続）
    new_env3 = SumoEnvironment(
        net_file=new_net3,
        route_file=new_route3,
        out_csv_name="outputs/baseline/dqn_after_switch",
        single_agent=True,
        use_gui=False,
        num_seconds=5000,
        print_step_progress=True,
    )

    model.set_env(new_env3)
    # If the new route matches the target, try to force the model's exploration rate to 0.5.
    def _try_set_model_exploration(m, value):
        
        names = [
            "exploration_rate",
            "exploration",
            "eps",
            "exploration_initial_eps",
        ]
        for n in names:
            try:
                if hasattr(m, n):
                    setattr(m, n, value)
                    print(f"Set model.{n} = {value}")
                    return True
            except Exception:
                pass

        # Try nested attributes commonly used by some implementations
        try:
            if hasattr(m, "policy") and hasattr(m.policy, "exploration"):
                try:
                    m.policy.exploration = value
                    print(f"Set model.policy.exploration = {value}")
                    return True
                except Exception:
                    pass
        except Exception:
            pass

        print("Could not set a recognized exploration attribute on the model; no-op")
        return False

    if os.path.basename(new_route3) == "baseline4_left_sn.rou.xml":
        # Try best-effort approaches to force epsilon to 0.5.
        _try_set_model_exploration(model, 0.5)

        # Stable-Baselines3 DQN uses a linear schedule based on
        # `exploration_initial_eps` and `exploration_final_eps`. To force
        # the instantaneous exploration rate to remain at 0.5, set both
        # endpoints to 0.5 so the schedule returns a constant value.
        changed = False
        if hasattr(model, "exploration_initial_eps"):
            try:
                model.exploration_initial_eps = 0.5
                # print("Set model.exploration_initial_eps = 0.5")
                print(f"Set model.exploration_initial_eps = {model.exploration_initial_eps}")
                print(f'Current exploration rate: {model.exploration_rate}')
                print(f'exploration_final_eps: {model.exploration_final_eps}')
                print(f'exploration_fraction: {model.exploration_fraction}')
                changed = True
            except Exception:
                pass
        # if hasattr(model, "exploration_final_eps"):
        #     try:
        #         model.exploration_final_eps = 0.5
        #         print("Set model.exploration_final_eps = 0.5")
        #         changed = True
        #     except Exception:
        #         pass

        # As a fallback, also try to set common policy-level attributes.
        # try:
        #     if hasattr(model, "policy"):
        #         if hasattr(model.policy, "exploration"):
        #             model.policy.exploration = 0.5
        #             print("Set model.policy.exploration = 0.5")
        #             changed = True
        #         if hasattr(model.policy, "epsilon"):
        #             model.policy.epsilon = 0.5
        #             print("Set model.policy.epsilon = 0.5")
        #             changed = True
        # except Exception:
        #     pass

        if not changed:
            print("Warning: could not set DQN schedule endpoints; exploration may continue to decay according to internal schedule")

    remaining = max(0, switch_at4 - switch_at3)
    if remaining > 0:
        model.learn(total_timesteps=remaining, reset_num_timesteps=False)

    new_net4 = "sumo_rl/nets/baseline/baseline3.net.xml"
    new_route4 = "sumo_rl/nets/baseline/baseline4_right_we.rou.xml"

    try:
        env.close()
    except Exception:
        pass


    # new_env4の開始
    # 新しい環境を作成してモデルに設定（学習状態は継続）
    new_env4 = SumoEnvironment(
        net_file=new_net4,
        route_file=new_route4,
        out_csv_name="outputs/baseline/dqn_after_switch",
        single_agent=True,
        use_gui=False,
        num_seconds=5000,
        print_step_progress=True,
    )

    model.set_env(new_env4)
    # If the new route matches the target, try to force the model's exploration rate to 0.5.
    def _try_set_model_exploration(m, value):
        
        names = [
            "exploration_rate",
            "exploration",
            "eps",
            "exploration_initial_eps",
        ]
        for n in names:
            try:
                if hasattr(m, n):
                    setattr(m, n, value)
                    print(f"Set model.{n} = {value}")
                    return True
            except Exception:
                pass

        # Try nested attributes commonly used by some implementations
        try:
            if hasattr(m, "policy") and hasattr(m.policy, "exploration"):
                try:
                    m.policy.exploration = value
                    print(f"Set model.policy.exploration = {value}")
                    return True
                except Exception:
                    pass
        except Exception:
            pass

        print("Could not set a recognized exploration attribute on the model; no-op")
        return False

    if os.path.basename(new_route4) == "baseline4_right_we.rou.xml":
        # Try best-effort approaches to force epsilon to 0.5.
        _try_set_model_exploration(model, 0.5)

        # Stable-Baselines3 DQN uses a linear schedule based on
        # `exploration_initial_eps` and `exploration_final_eps`. To force
        # the instantaneous exploration rate to remain at 0.5, set both
        # endpoints to 0.5 so the schedule returns a constant value.
        changed = False
        if hasattr(model, "exploration_initial_eps"):
            try:
                model.exploration_initial_eps = 0.5
                # print("Set model.exploration_initial_eps = 0.5")
                print(f"Set model.exploration_initial_eps = {model.exploration_initial_eps}")
                print(f'Current exploration rate: {model.exploration_rate}')
                print(f'exploration_final_eps: {model.exploration_final_eps}')
                print(f'exploration_fraction: {model.exploration_fraction}')
                changed = True
            except Exception:
                pass
        # if hasattr(model, "exploration_final_eps"):
        #     try:
        #         model.exploration_final_eps = 0.5
        #         print("Set model.exploration_final_eps = 0.5")
        #         changed = True
        #     except Exception:
        #         pass

        # As a fallback, also try to set common policy-level attributes.
        # try:
        #     if hasattr(model, "policy"):
        #         if hasattr(model.policy, "exploration"):
        #             model.policy.exploration = 0.5
        #             print("Set model.policy.exploration = 0.5")
        #             changed = True
        #         if hasattr(model.policy, "epsilon"):
        #             model.policy.epsilon = 0.5
        #             print("Set model.policy.epsilon = 0.5")
        #             changed = True
        # except Exception:
        #     pass

        if not changed:
            print("Warning: could not set DQN schedule endpoints; exploration may continue to decay according to internal schedule")

    remaining = max(0, switch_at5 - switch_at4)
    if remaining > 0:
        model.learn(total_timesteps=remaining, reset_num_timesteps=False)

    new_net5 = "sumo_rl/nets/baseline/baseline3.net.xml"
    new_route5 = "sumo_rl/nets/baseline/baseline4_right_sn.rou.xml"

    try:
        env.close()
    except Exception:
        pass


    # new_env5の開始
    # 新しい環境を作成してモデルに設定（学習状態は継続）
    new_env5 = SumoEnvironment(
        net_file=new_net5,
        route_file=new_route5,
        out_csv_name="outputs/baseline/dqn_after_switch",
        single_agent=True,
        use_gui=False,
        num_seconds=5000,
        print_step_progress=True,
    )

    model.set_env(new_env5)
    # If the new route matches the target, try to force the model's exploration rate to 0.5.
    def _try_set_model_exploration(m, value):
        
        names = [
            "exploration_rate",
            "exploration",
            "eps",
            "exploration_initial_eps",
        ]
        for n in names:
            try:
                if hasattr(m, n):
                    setattr(m, n, value)
                    print(f"Set model.{n} = {value}")
                    return True
            except Exception:
                pass

        # Try nested attributes commonly used by some implementations
        try:
            if hasattr(m, "policy") and hasattr(m.policy, "exploration"):
                try:
                    m.policy.exploration = value
                    print(f"Set model.policy.exploration = {value}")
                    return True
                except Exception:
                    pass
        except Exception:
            pass

        print("Could not set a recognized exploration attribute on the model; no-op")
        return False

    if os.path.basename(new_route5) == "baseline4_right_sn.rou.xml":
        # Try best-effort approaches to force epsilon to 0.5.
        _try_set_model_exploration(model, 0.5)

        # Stable-Baselines3 DQN uses a linear schedule based on
        # `exploration_initial_eps` and `exploration_final_eps`. To force
        # the instantaneous exploration rate to remain at 0.5, set both
        # endpoints to 0.5 so the schedule returns a constant value.
        changed = False
        if hasattr(model, "exploration_initial_eps"):
            try:
                model.exploration_initial_eps = 0.5
                # print("Set model.exploration_initial_eps = 0.5")
                print(f"Set model.exploration_initial_eps = {model.exploration_initial_eps}")
                print(f'Current exploration rate: {model.exploration_rate}')
                print(f'exploration_final_eps: {model.exploration_final_eps}')
                print(f'exploration_fraction: {model.exploration_fraction}')
                changed = True
            except Exception:
                pass
        # if hasattr(model, "exploration_final_eps"):
        #     try:
        #         model.exploration_final_eps = 0.5
        #         print("Set model.exploration_final_eps = 0.5")
        #         changed = True
        #     except Exception:
        #         pass

        # As a fallback, also try to set common policy-level attributes.
        # try:
        #     if hasattr(model, "policy"):
        #         if hasattr(model.policy, "exploration"):
        #             model.policy.exploration = 0.5
        #             print("Set model.policy.exploration = 0.5")
        #             changed = True
        #         if hasattr(model.policy, "epsilon"):
        #             model.policy.epsilon = 0.5
        #             print("Set model.policy.epsilon = 0.5")
        #             changed = True
        # except Exception:
        #     pass

        if not changed:
            print("Warning: could not set DQN schedule endpoints; exploration may continue to decay according to internal schedule")

    remaining = max(0, switch_at6 - switch_at5)
    if remaining > 0:
        model.learn(total_timesteps=remaining, reset_num_timesteps=False)

    new_net6 = "sumo_rl/nets/baseline/baseline3.net.xml"
    new_route6 = "sumo_rl/nets/baseline/baseline4_straight.rou.xml"

    try:
        env.close()
    except Exception:
        pass


    # new_env6の開始
    # 新しい環境を作成してモデルに設定（学習状態は継続）
    new_env6 = SumoEnvironment(
        net_file=new_net6,
        route_file=new_route6,
        out_csv_name="outputs/baseline/dqn_after_switch",
        single_agent=True,
        use_gui=False,
        num_seconds=5000,
        print_step_progress=True,
    )

    model.set_env(new_env6)
    # If the new route matches the target, try to force the model's exploration rate to 0.5.
    def _try_set_model_exploration(m, value):
        
        names = [
            "exploration_rate",
            "exploration",
            "eps",
            "exploration_initial_eps",
        ]
        for n in names:
            try:
                if hasattr(m, n):
                    setattr(m, n, value)
                    print(f"Set model.{n} = {value}")
                    return True
            except Exception:
                pass

        # Try nested attributes commonly used by some implementations
        try:
            if hasattr(m, "policy") and hasattr(m.policy, "exploration"):
                try:
                    m.policy.exploration = value
                    print(f"Set model.policy.exploration = {value}")
                    return True
                except Exception:
                    pass
        except Exception:
            pass

        print("Could not set a recognized exploration attribute on the model; no-op")
        return False

    if os.path.basename(new_route6) == "baseline4_straight.rou.xml":
        # Try best-effort approaches to force epsilon to 0.5.
        _try_set_model_exploration(model, 0.5)

        # Stable-Baselines3 DQN uses a linear schedule based on
        # `exploration_initial_eps` and `exploration_final_eps`. To force
        # the instantaneous exploration rate to remain at 0.5, set both
        # endpoints to 0.5 so the schedule returns a constant value.
        changed = False
        if hasattr(model, "exploration_initial_eps"):
            try:
                model.exploration_initial_eps = 0.5
                # print("Set model.exploration_initial_eps = 0.5")
                print(f"Set model.exploration_initial_eps = {model.exploration_initial_eps}")
                print(f'Current exploration rate: {model.exploration_rate}')
                print(f'exploration_final_eps: {model.exploration_final_eps}')
                print(f'exploration_fraction: {model.exploration_fraction}')
                changed = True
            except Exception:
                pass
        # if hasattr(model, "exploration_final_eps"):
        #     try:
        #         model.exploration_final_eps = 0.5
        #         print("Set model.exploration_final_eps = 0.5")
        #         changed = True
        #     except Exception:
        #         pass

        # As a fallback, also try to set common policy-level attributes.
        # try:
        #     if hasattr(model, "policy"):
        #         if hasattr(model.policy, "exploration"):
        #             model.policy.exploration = 0.5
        #             print("Set model.policy.exploration = 0.5")
        #             changed = True
        #         if hasattr(model.policy, "epsilon"):
        #             model.policy.epsilon = 0.5
        #             print("Set model.policy.epsilon = 0.5")
        #             changed = True
        # except Exception:
        #     pass

        if not changed:
            print("Warning: could not set DQN schedule endpoints; exploration may continue to decay according to internal schedule")

    remaining = max(0, switch_at7 - switch_at6)
    if remaining > 0:
        model.learn(total_timesteps=remaining, reset_num_timesteps=False)

    new_net7 = "sumo_rl/nets/baseline/baseline3.net.xml"
    new_route7 = "sumo_rl/nets/baseline/baseline4_left.rou.xml"

    try:
        env.close()
    except Exception:
        pass


    # new_env7の開始
    # 新しい環境を作成してモデルに設定（学習状態は継続）
    new_env7 = SumoEnvironment(
        net_file=new_net7,
        route_file=new_route7,
        out_csv_name="outputs/baseline/dqn_after_switch",
        single_agent=True,
        use_gui=False,
        num_seconds=5000,
        print_step_progress=True,
    )

    model.set_env(new_env7)
    # If the new route matches the target, try to force the model's exploration rate to 0.5.
    def _try_set_model_exploration(m, value):
        
        names = [
            "exploration_rate",
            "exploration",
            "eps",
            "exploration_initial_eps",
        ]
        for n in names:
            try:
                if hasattr(m, n):
                    setattr(m, n, value)
                    print(f"Set model.{n} = {value}")
                    return True
            except Exception:
                pass

        # Try nested attributes commonly used by some implementations
        try:
            if hasattr(m, "policy") and hasattr(m.policy, "exploration"):
                try:
                    m.policy.exploration = value
                    print(f"Set model.policy.exploration = {value}")
                    return True
                except Exception:
                    pass
        except Exception:
            pass

        print("Could not set a recognized exploration attribute on the model; no-op")
        return False

    if os.path.basename(new_route7) == "baseline4_left.rou.xml":
        # Try best-effort approaches to force epsilon to 0.5.
        _try_set_model_exploration(model, 0.5)

        # Stable-Baselines3 DQN uses a linear schedule based on
        # `exploration_initial_eps` and `exploration_final_eps`. To force
        # the instantaneous exploration rate to remain at 0.5, set both
        # endpoints to 0.5 so the schedule returns a constant value.
        changed = False
        if hasattr(model, "exploration_initial_eps"):
            try:
                model.exploration_initial_eps = 0.5
                # print("Set model.exploration_initial_eps = 0.5")
                print(f"Set model.exploration_initial_eps = {model.exploration_initial_eps}")
                print(f'Current exploration rate: {model.exploration_rate}')
                print(f'exploration_final_eps: {model.exploration_final_eps}')
                print(f'exploration_fraction: {model.exploration_fraction}')
                changed = True
            except Exception:
                pass
        # if hasattr(model, "exploration_final_eps"):
        #     try:
        #         model.exploration_final_eps = 0.5
        #         print("Set model.exploration_final_eps = 0.5")
        #         changed = True
        #     except Exception:
        #         pass

        # As a fallback, also try to set common policy-level attributes.
        # try:
        #     if hasattr(model, "policy"):
        #         if hasattr(model.policy, "exploration"):
        #             model.policy.exploration = 0.5
        #             print("Set model.policy.exploration = 0.5")
        #             changed = True
        #         if hasattr(model.policy, "epsilon"):
        #             model.policy.epsilon = 0.5
        #             print("Set model.policy.epsilon = 0.5")
        #             changed = True
        # except Exception:
        #     pass

        if not changed:
            print("Warning: could not set DQN schedule endpoints; exploration may continue to decay according to internal schedule")

    remaining = max(0, switch_at8 - switch_at7)
    if remaining > 0:
        model.learn(total_timesteps=remaining, reset_num_timesteps=False)

    new_net8 = "sumo_rl/nets/baseline/baseline3.net.xml"
    new_route8 = "sumo_rl/nets/baseline/baseline4_we.rou.xml"

    try:
        env.close()
    except Exception:
        pass


    # new_env8の開始
    # 新しい環境を作成してモデルに設定（学習状態は継続）
    new_env8 = SumoEnvironment(
        net_file=new_net8,
        route_file=new_route8,
        out_csv_name="outputs/baseline/dqn_after_switch",
        single_agent=True,
        use_gui=False,
        num_seconds=5000,
        print_step_progress=True,
    )

    model.set_env(new_env8)
    # If the new route matches the target, try to force the model's exploration rate to 0.5.
    def _try_set_model_exploration(m, value):
        
        names = [
            "exploration_rate",
            "exploration",
            "eps",
            "exploration_initial_eps",
        ]
        for n in names:
            try:
                if hasattr(m, n):
                    setattr(m, n, value)
                    print(f"Set model.{n} = {value}")
                    return True
            except Exception:
                pass

        # Try nested attributes commonly used by some implementations
        try:
            if hasattr(m, "policy") and hasattr(m.policy, "exploration"):
                try:
                    m.policy.exploration = value
                    print(f"Set model.policy.exploration = {value}")
                    return True
                except Exception:
                    pass
        except Exception:
            pass

        print("Could not set a recognized exploration attribute on the model; no-op")
        return False

    if os.path.basename(new_route8) == "baseline4_we.rou.xml":
        # Try best-effort approaches to force epsilon to 0.5.
        _try_set_model_exploration(model, 0.5)

        # Stable-Baselines3 DQN uses a linear schedule based on
        # `exploration_initial_eps` and `exploration_final_eps`. To force
        # the instantaneous exploration rate to remain at 0.5, set both
        # endpoints to 0.5 so the schedule returns a constant value.
        changed = False
        if hasattr(model, "exploration_initial_eps"):
            try:
                model.exploration_initial_eps = 0.5
                # print("Set model.exploration_initial_eps = 0.5")
                print(f"Set model.exploration_initial_eps = {model.exploration_initial_eps}")
                print(f'Current exploration rate: {model.exploration_rate}')
                print(f'exploration_final_eps: {model.exploration_final_eps}')
                print(f'exploration_fraction: {model.exploration_fraction}')
                changed = True
            except Exception:
                pass
        # if hasattr(model, "exploration_final_eps"):
        #     try:
        #         model.exploration_final_eps = 0.5
        #         print("Set model.exploration_final_eps = 0.5")
        #         changed = True
        #     except Exception:
        #         pass

        # As a fallback, also try to set common policy-level attributes.
        # try:
        #     if hasattr(model, "policy"):
        #         if hasattr(model.policy, "exploration"):
        #             model.policy.exploration = 0.5
        #             print("Set model.policy.exploration = 0.5")
        #             changed = True
        #         if hasattr(model.policy, "epsilon"):
        #             model.policy.epsilon = 0.5
        #             print("Set model.policy.epsilon = 0.5")
        #             changed = True
        # except Exception:
        #     pass

        if not changed:
            print("Warning: could not set DQN schedule endpoints; exploration may continue to decay according to internal schedule")

    remaining = max(0, switch_at9 - switch_at8)
    if remaining > 0:
        model.learn(total_timesteps=remaining, reset_num_timesteps=False)

    new_net9 = "sumo_rl/nets/baseline/baseline3.net.xml"
    new_route9 = "sumo_rl/nets/baseline/baseline4_sn.rou.xml"

    try:
        env.close()
    except Exception:
        pass


    # new_env9の開始
    # 新しい環境を作成してモデルに設定（学習状態は継続）
    new_env9 = SumoEnvironment(
        net_file=new_net9,
        route_file=new_route9,
        out_csv_name="outputs/baseline/dqn_after_switch",
        single_agent=True,
        use_gui=False,
        num_seconds=5000,
        print_step_progress=True,
    )

    model.set_env(new_env9)
    # If the new route matches the target, try to force the model's exploration rate to 0.5.
    def _try_set_model_exploration(m, value):
        
        names = [
            "exploration_rate",
            "exploration",
            "eps",
            "exploration_initial_eps",
        ]
        for n in names:
            try:
                if hasattr(m, n):
                    setattr(m, n, value)
                    print(f"Set model.{n} = {value}")
                    return True
            except Exception:
                pass

        # Try nested attributes commonly used by some implementations
        try:
            if hasattr(m, "policy") and hasattr(m.policy, "exploration"):
                try:
                    m.policy.exploration = value
                    print(f"Set model.policy.exploration = {value}")
                    return True
                except Exception:
                    pass
        except Exception:
            pass

        print("Could not set a recognized exploration attribute on the model; no-op")
        return False

    if os.path.basename(new_route9) == "baseline4_sn.rou.xml":
        # Try best-effort approaches to force epsilon to 0.5.
        _try_set_model_exploration(model, 0.5)

        # Stable-Baselines3 DQN uses a linear schedule based on
        # `exploration_initial_eps` and `exploration_final_eps`. To force
        # the instantaneous exploration rate to remain at 0.5, set both
        # endpoints to 0.5 so the schedule returns a constant value.
        changed = False
        if hasattr(model, "exploration_initial_eps"):
            try:
                model.exploration_initial_eps = 0.5
                # print("Set model.exploration_initial_eps = 0.5")
                print(f"Set model.exploration_initial_eps = {model.exploration_initial_eps}")
                print(f'Current exploration rate: {model.exploration_rate}')
                print(f'exploration_final_eps: {model.exploration_final_eps}')
                print(f'exploration_fraction: {model.exploration_fraction}')
                changed = True
            except Exception:
                pass
        # if hasattr(model, "exploration_final_eps"):
        #     try:
        #         model.exploration_final_eps = 0.5
        #         print("Set model.exploration_final_eps = 0.5")
        #         changed = True
        #     except Exception:
        #         pass

        # As a fallback, also try to set common policy-level attributes.
        # try:
        #     if hasattr(model, "policy"):
        #         if hasattr(model.policy, "exploration"):
        #             model.policy.exploration = 0.5
        #             print("Set model.policy.exploration = 0.5")
        #             changed = True
        #         if hasattr(model.policy, "epsilon"):
        #             model.policy.epsilon = 0.5
        #             print("Set model.policy.epsilon = 0.5")
        #             changed = True
        # except Exception:
        #     pass

        if not changed:
            print("Warning: could not set DQN schedule endpoints; exploration may continue to decay according to internal schedule")

    remaining = max(0, switch_at10 - switch_at9)
    if remaining > 0:
        model.learn(total_timesteps=remaining, reset_num_timesteps=False)

    new_net10 = "sumo_rl/nets/baseline/baseline3.net.xml"
    new_route10 = "sumo_rl/nets/baseline/baseline4.rou.xml"

    try:
        env.close()
    except Exception:
        pass


    # new_env10の開始
    # 新しい環境を作成してモデルに設定（学習状態は継続）
    new_env10 = SumoEnvironment(
        net_file=new_net10,
        route_file=new_route10,
        out_csv_name="outputs/baseline/dqn_after_switch",
        single_agent=True,
        use_gui=False,
        num_seconds=5000,
        print_step_progress=True,
    )

    model.set_env(new_env10)
    # If the new route matches the target, try to force the model's exploration rate to 0.5.
    def _try_set_model_exploration(m, value):
        
        names = [
            "exploration_rate",
            "exploration",
            "eps",
            "exploration_initial_eps",
        ]
        for n in names:
            try:
                if hasattr(m, n):
                    setattr(m, n, value)
                    print(f"Set model.{n} = {value}")
                    return True
            except Exception:
                pass

        # Try nested attributes commonly used by some implementations
        try:
            if hasattr(m, "policy") and hasattr(m.policy, "exploration"):
                try:
                    m.policy.exploration = value
                    print(f"Set model.policy.exploration = {value}")
                    return True
                except Exception:
                    pass
        except Exception:
            pass

        print("Could not set a recognized exploration attribute on the model; no-op")
        return False

    if os.path.basename(new_route10) == "baseline4.rou.xml":
        # Try best-effort approaches to force epsilon to 0.5.
        _try_set_model_exploration(model, 0.2)

        # Stable-Baselines3 DQN uses a linear schedule based on
        # `exploration_initial_eps` and `exploration_final_eps`. To force
        # the instantaneous exploration rate to remain at 0.5, set both
        # endpoints to 0.5 so the schedule returns a constant value.
        changed = False
        if hasattr(model, "exploration_initial_eps"):
            try:
                model.exploration_initial_eps = 0.2
                # print("Set model.exploration_initial_eps = 0.2")
                print(f"Set model.exploration_initial_eps = {model.exploration_initial_eps}")
                print(f'Current exploration rate: {model.exploration_rate}')
                print(f'exploration_final_eps: {model.exploration_final_eps}')
                print(f'exploration_fraction: {model.exploration_fraction}')
                changed = True
            except Exception:
                pass
        # if hasattr(model, "exploration_final_eps"):
        #     try:
        #         model.exploration_final_eps = 0.2
        #         print("Set model.exploration_final_eps = 0.2")
        #         changed = True
        #     except Exception:
        #         pass

        # As a fallback, also try to set common policy-level attributes.
        # try:
        #     if hasattr(model, "policy"):
        #         if hasattr(model.policy, "exploration"):
        #             model.policy.exploration = 0.2
        #             print("Set model.policy.exploration = 0.2")
        #             changed = True
        #         if hasattr(model.policy, "epsilon"):
        #             model.policy.epsilon = 0.2
        #             print("Set model.policy.epsilon = 0.2")
        #             changed = True
        # except Exception:
        #     pass

        if not changed:
            print("Warning: could not set DQN schedule endpoints; exploration may continue to decay according to internal schedule")

    remaining = max(0, switch_at11 - switch_at10)
    if remaining > 0:
        model.learn(total_timesteps=remaining, reset_num_timesteps=False)

    new_net11 = "sumo_rl/nets/baseline/baseline3.net.xml"
    new_route11 = "sumo_rl/nets/baseline/baseline5.rou.xml"

    try:
        env.close()
    except Exception:
        pass


    # new_env11の開始
    # 新しい環境を作成してモデルに設定（学習状態は継続）
    new_env11 = SumoEnvironment(
        net_file=new_net11,
        route_file=new_route11,
        out_csv_name="outputs/baseline/dqn_after_switch",
        single_agent=True,
        use_gui=False,
        num_seconds=5000,
        print_step_progress=True,
    )

    model.set_env(new_env11)
    # If the new route matches the target, try to force the model's exploration rate to 0.5.
    def _try_set_model_exploration(m, value):
        
        names = [
            "exploration_rate",
            "exploration",
            "eps",
            "exploration_initial_eps",
        ]
        for n in names:
            try:
                if hasattr(m, n):
                    setattr(m, n, value)
                    print(f"Set model.{n} = {value}")
                    return True
            except Exception:
                pass

        # Try nested attributes commonly used by some implementations
        try:
            if hasattr(m, "policy") and hasattr(m.policy, "exploration"):
                try:
                    m.policy.exploration = value
                    print(f"Set model.policy.exploration = {value}")
                    return True
                except Exception:
                    pass
        except Exception:
            pass

        print("Could not set a recognized exploration attribute on the model; no-op")
        return False

    if os.path.basename(new_route11) == "baseline5.rou.xml":
        # Try best-effort approaches to force epsilon to 0.5.
        _try_set_model_exploration(model, 0.2)

        # Stable-Baselines3 DQN uses a linear schedule based on
        # `exploration_initial_eps` and `exploration_final_eps`. To force
        # the instantaneous exploration rate to remain at 0.5, set both
        # endpoints to 0.5 so the schedule returns a constant value.
        changed = False
        if hasattr(model, "exploration_initial_eps"):
            try:
                model.exploration_initial_eps = 0.2
                # print("Set model.exploration_initial_eps = 0.2")
                print(f"Set model.exploration_initial_eps = {model.exploration_initial_eps}")
                print(f'Current exploration rate: {model.exploration_rate}')
                print(f'exploration_final_eps: {model.exploration_final_eps}')
                print(f'exploration_fraction: {model.exploration_fraction}')
                changed = True
            except Exception:
                pass
        # if hasattr(model, "exploration_final_eps"):
        #     try:
        #         model.exploration_final_eps = 0.2
        #         print("Set model.exploration_final_eps = 0.2")
        #         changed = True
        #     except Exception:
        #         pass

        # As a fallback, also try to set common policy-level attributes.
        # try:
        #     if hasattr(model, "policy"):
        #         if hasattr(model.policy, "exploration"):
        #             model.policy.exploration = 0.2
        #             print("Set model.policy.exploration = 0.2")
        #             changed = True
        #         if hasattr(model.policy, "epsilon"):
        #             model.policy.epsilon = 0.2
        #             print("Set model.policy.epsilon = 0.2")
        #             changed = True
        # except Exception:
        #     pass

        if not changed:
            print("Warning: could not set DQN schedule endpoints; exploration may continue to decay according to internal schedule")

    remaining = max(0, switch_at12 - switch_at11)
    if remaining > 0:
        model.learn(total_timesteps=remaining, reset_num_timesteps=False)

    new_net12 = "sumo_rl/nets/baseline/baseline3.net.xml"
    new_route12 = "sumo_rl/nets/baseline/baseline6.rou.xml"

    try:
        env.close()
    except Exception:
        pass

    # new_env12の開始 
    # 新しい環境を作成してモデルに設定（学習状態は継続）
    new_env = SumoEnvironment(
        net_file=new_net12,
        route_file=new_route12,
        out_csv_name="outputs/baseline/dqn_after_switch",
        single_agent=True,
        use_gui=False,
        num_seconds=5000,
        print_step_progress=True,
    )

    # Wrapper to randomize the route file at every episode reset

    class RandomizeRouteOnReset(gym.Wrapper):
        def __init__(self, env, min_prob=0.05, max_prob=0.3, seed_base=10, precision=4):
            super().__init__(env)
            self.min_prob = min_prob
            self.max_prob = max_prob
            self.seed_base = seed_base
            self.precision = precision
            self._local_episode = 0

        def reset(self, *args, **kwargs):
            # derive a seed per-episode if seed_base given, else None
            seed = None
            if self.seed_base is not None:
                seed = (self.seed_base + self._local_episode) if isinstance(self.seed_base, int) else None

            try:
                # randomize in-place (overwrite the route file)
                randomize_flow_probability(self.env._route, None, self.min_prob, self.max_prob, seed, self.precision)
            except Exception as e:
                print(f"Warning: could not randomize route file {getattr(self.env, '_route', None)}: {e}")

            self._local_episode += 1
            return self.env.reset(*args, **kwargs)

    # Wrap the new environment so that every episode its route file is randomized

    new_env = RandomizeRouteOnReset(new_env, min_prob=0.05, max_prob=0.3, seed_base=None)

    # Stable-Baselines3 に環境を差し替え（タイムステップはリセットしない）
    
    model.set_env(new_env)

    def _try_set_model_exploration(m, value):
        
        names = [
            "exploration_rate",
            "exploration",
            "eps",
            "exploration_initial_eps",
        ]
        for n in names:
            try:
                if hasattr(m, n):
                    setattr(m, n, value)
                    print(f"Set model.{n} = {value}")
                    return True
            except Exception:
                pass

        # Try nested attributes commonly used by some implementations
        try:
            if hasattr(m, "policy") and hasattr(m.policy, "exploration"):
                try:
                    m.policy.exploration = value
                    print(f"Set model.policy.exploration = {value}")
                    return True
                except Exception:
                    pass
        except Exception:
            pass

        print("Could not set a recognized exploration attribute on the model; no-op")
        return False

    if os.path.basename(new_route12) == "baseline6.rou.xml":
        # Try best-effort approaches to force epsilon to 0.3.
        _try_set_model_exploration(model, 0.2)

        # Stable-Baselines3 DQN uses a linear schedule based on
        # `exploration_initial_eps` and `exploration_final_eps`. To force
        # the instantaneous exploration rate to remain at 0.5, set both
        # endpoints to 0.5 so the schedule returns a constant value.
        changed = False
        if hasattr(model, "exploration_initial_eps"):
            try:
                model.exploration_initial_eps = 0.2
                # print("Set model.exploration_initial_eps = 0.3")
                print(f"Set model.exploration_initial_eps = {model.exploration_initial_eps}")
                print(f'Current exploration rate: {model.exploration_rate}')
                print(f'exploration_final_eps: {model.exploration_final_eps}')
                print(f'exploration_fraction: {model.exploration_fraction}')
                changed = True
            except Exception:
                pass
        # if hasattr(model, "exploration_final_eps"):
        #     try:
        #         model.exploration_final_eps = 0.5
        #         print("Set model.exploration_final_eps = 0.5")
        #         changed = True
        #     except Exception:
        #         pass

        # As a fallback, also try to set common policy-level attributes.
        # try:
        #     if hasattr(model, "policy"):
        #         if hasattr(model.policy, "exploration"):
        #             model.policy.exploration = 0.5
        #             print("Set model.policy.exploration = 0.5")
        #             changed = True
        #         if hasattr(model.policy, "epsilon"):
        #             model.policy.epsilon = 0.5
        #             print("Set model.policy.epsilon = 0.5")
        #             changed = True
        # except Exception:
        #     pass

        if not changed:
            print("Warning: could not set DQN schedule endpoints; exploration may continue to decay according to internal schedule")

    remaining = max(0, total_timesteps - switch_at12)
    if remaining > 0:
        model.learn(total_timesteps=remaining, reset_num_timesteps=False)

    # モデルを保存
    model_save_path = "outputs/baseline/baseline5_6_dqn_model"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")
