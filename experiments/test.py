import os
import sys
import argparse
import random
import gymnasium as gym
from stable_baselines3.dqn.dqn import DQN
from pathlib import Path

# ローカルリポジトリを優先的に読み込む
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import traci

import sumo_rl
from sumo_rl import SumoEnvironment
from scripts.randomize_flow_probability import randomize_flow_probability
import shutil

if __name__ == "__main__":
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description="Test DQN model on baseline3")
    parser.add_argument("--model-path", type=str, default="outputs/baseline/baseline3_dqn_model",
                        help="Path to the trained model (without .zip extension)")
    parser.add_argument("--num-episodes", type=int, default=50,
                        help="Number of episodes to evaluate")
    parser.add_argument("--use-gui", action="store_true",
                        help="Use SUMO GUI for visualization")
    parser.add_argument("--randomize-route", action="store_true",
                        help="Randomize <flow>@probability on each episode reset")
    parser.add_argument("--min-prob", type=float, default=0.05,
                        help="Minimum flow probability when randomizing")
    parser.add_argument("--max-prob", type=float, default=0.3,
                        help="Maximum flow probability when randomizing")
    parser.add_argument("--seed-base", type=int, default=None,
                        help="Optional base seed for deterministic per-episode randomization")
    parser.add_argument("--change-destination", action="store_true",
                        help="Enable destination change near intersections")
    parser.add_argument("--change-distance", type=float, default=10.0,
                        help="Distance (m) from lane end to trigger destination change")
    parser.add_argument("--change-probability", type=float, default=0.1,
                        help="Probability to apply destination change")
    args = parser.parse_args()

    env = SumoEnvironment(
        net_file="sumo_rl/nets/baseline/baseline3.net.xml",
        route_file="sumo_rl/nets/baseline/baseline6.rou.xml",
        out_csv_name="outputs/baseline/dqn_test",
        single_agent=True,
        use_gui=args.use_gui,  # コマンドライン引数から指定
        num_seconds=5000,
    )

    # オプション: エピソードごとに probability をランダム化する設定
    # コマンドライン引数で制御できるようにする
    seed_base = args.seed_base
    min_prob = args.min_prob
    max_prob = args.max_prob
    do_randomize = args.randomize_route

    # モデルのロード
    model_path = args.model_path
    if not os.path.exists(model_path + ".zip"):
        raise FileNotFoundError(f"Model file not found: {model_path}.zip")
    model = DQN.load(model_path)
    print(f"Model loaded from {model_path}")

    # 50エピソード評価
    num_episodes = args.num_episodes
    rewards = []
    arrived_vehicles = []
    waiting_times = []
    co2_emission = []
    for ep in range(num_episodes):
        # # エピソードごとにルートファイルを切り替える
        # # 例: sumo_rl/nets/baseline/baseline6_ep{ep}.rou.xml
        episode_route = os.path.join("sumo_rl", "nets", "baseline", f"baseline6_ep{ep}.rou.xml")
        if os.path.exists(episode_route):
            # SumoEnvironment では内部で _route や route_file を使っていることがあるため両方を差し替える
            try:
                env._route = episode_route
            except Exception:
                pass
            try:
                env.route_file = episode_route
            except Exception:
                pass
            print(f"[EpisodeRoute] ep={ep} using route_file={episode_route}")
        else:
            print(f"[EpisodeRoute] ep={ep} route file not found: {episode_route} — using existing route {getattr(env, '_route', getattr(env, 'route_file', 'unknown'))}")

        # # エピソード開始前にルートをランダム化して一時ファイルを作り、env._route を差し替える（ランダム化が有効な場合）
        # if do_randomize:
        #     orig_route = env._route
        #     seed = (seed_base + ep) if seed_base is not None else None
        #     try:
        #         # 出力先ディレクトリを作成し、エピソードごとのファイル名を作成
        #         save_dir = os.path.join("outputs")
        #         os.makedirs(save_dir, exist_ok=True)
        #         # ユーザー要望に合わせ、固定の命名規則にする
        #         filename = f"baseline6_ep{ep}.rou.xml"
        #         dest_route = os.path.join(save_dir, filename)

        #         # 元のルートを保持したまま、ランダム化結果を別ファイルに保存
        #         randomize_flow_probability(orig_route, dest_route, min_prob, max_prob, seed, precision=4)

        #         # 環境がこのエピソードで新しいルートを使うように差し替え
        #         try:
        #             env._route = dest_route
        #         except Exception:
        #             pass
        #         try:
        #             env.route_file = dest_route
        #         except Exception:
        #             pass

        #         # SUMO のシードも合わせる（SumoEnvironment は reset 時に self.sumo_seed を参照する）
        #         if seed is not None:
        #             try:
        #                 env.sumo_seed = seed
        #             except Exception:
        #                 pass
        #         print(f"[RouteRandom] ep={ep} seed={seed} saved_route={dest_route}")
        #     except Exception as e:
        #         print(f"Warning: could not randomize route for ep {ep}: {e}")

        obs, info = env.reset()
        done = False
        total_reward = 0
        step = 0
        cumulative_waiting_time = 0  # 累積待機時間 (システム全体のステップごとの合計を加算)
        cumulative_arrived = 0  # エピソード中に到着した車両の総数 (最後の値を使用)
        cumulative_departed = 0
        changed_vehicles = set()  # 目的地変更済み車両の追跡
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1

            # 交差点直前で特定の車両の目的地を変更（オプション）
            if args.change_destination:
                try:
                    for veh_id in traci.vehicle.getIDList():
                        if veh_id in changed_vehicles:
                            continue

                        # 確率的に適用
                        if random.random() > args.change_probability:
                            continue

                        # 走行距離が130m〜150mの車両のみ対象
                        try:
                            dist = traci.vehicle.getDistance(veh_id)
                        except traci.exceptions.TraCIException:
                            continue
                        if not (130.0 <= dist < 150.0):
                            continue

                        # 出発点の車両（ルートインデックス=0）を対象に経路変更
                        try:
                            route_idx = traci.vehicle.getRouteIndex(veh_id)
                        except traci.exceptions.TraCIException:
                            route_idx = -1
                        # if route_idx == 0:
                        if 130.0 <= dist < 149.0:

                            current_route = traci.vehicle.getRoute(veh_id)
                            if len(current_route) == 0:
                                continue
                            # 現在位置のエッジ（発進エッジ）
                            origin_edge = current_route[0]
                            current_destination = current_route[-1]

                            # 全ての可能な目的地
                            all_targets = ["t_s", "t_e", "t_n", "t_w"]
                            
                            # 出発エッジの方角を判定し、その方角の目的地を除外
                            excluded_by_origin = None
                            if origin_edge.startswith("w"):
                                excluded_by_origin = "t_w"
                            elif origin_edge.startswith("s"):
                                excluded_by_origin = "t_s"
                            elif origin_edge.startswith("n"):
                                excluded_by_origin = "t_n"
                            elif origin_edge.startswith("e"):
                                excluded_by_origin = "t_e"
                            
                            # 現在の目的地と出発エッジと同じ方角の目的地を除外
                            candidates = [t for t in all_targets 
                                         if t != current_destination and t != excluded_by_origin]

                            new_destination = random.choice(candidates)
                            if new_destination == origin_edge:
                                continue
                            try:
                                route_obj = traci.simulation.findRoute(origin_edge, new_destination)
                                if route_obj and route_obj.edges:
                                    traci.vehicle.setRoute(veh_id, route_obj.edges)
                                    changed_vehicles.add(veh_id)
                                    # print(f"[DestChange] ep={ep+1} step={step} veh={veh_id} {current_destination}->{new_destination}")
                            except traci.exceptions.TraCIException:
                                print(f"[DestChange] route change failed for veh={veh_id}")
                except Exception as e:
                    print(f"[DestChange] error: {e}")
            # infoからwaiting timeを取得し累積
            if "agents_total_accumulated_waiting_time" in info:
                cumulative_waiting_time += info["agents_total_accumulated_waiting_time"]
            # 到着車両数は system_total_arrived に保存されている
            if "system_total_departed" in info:
                cumulative_departed = info["system_total_departed"]
            if "system_total_arrived" in info:
                cumulative_arrived = info["system_total_arrived"]
            if "system_total_co2_emission" in info:
                total_co2_emission = info["system_total_co2_emission"]
            done = terminated or truncated
        rewards.append(total_reward)
        # 1台あたりの累積待機時間を計算
        if cumulative_departed > 0:
            waiting_per_vehicle = cumulative_waiting_time / cumulative_departed
            co2_emission_per_vehicle = total_co2_emission / cumulative_departed
        else:
            waiting_per_vehicle = float('nan')
        arrived_vehicles.append(cumulative_arrived)
        waiting_times.append(waiting_per_vehicle)
        co2_emission.append(co2_emission_per_vehicle)
        print(f"Episode {ep+1}: Total reward = {total_reward}")
        print(f"total_arrived vehicles = {cumulative_arrived}")
        print(f"Cumulative waiting time (per vehicle) = {waiting_per_vehicle}")
        print(f"total_co2_emission = {total_co2_emission}")
        print(f"co2_emission per vehicle = {co2_emission_per_vehicle}")
        print("\n")
    print(f"\nwaiting_times: {waiting_times}")
    print(f"\nAverage reward over {num_episodes} episodes: {sum(rewards)/num_episodes}")
    print(f"Average arrived vehicles over {num_episodes} episodes: {sum(arrived_vehicles)/num_episodes}")
    print(f"Average cumulative waiting time over {num_episodes} episodes: {sum(waiting_times)/num_episodes}")
    print(f"Avarage total_co2_emission over {num_episodes} episodes: {sum(co2_emission)/num_episodes}")
