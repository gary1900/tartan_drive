import argparse
import os
import random

import torch


def load_trajs(data_fp, device):
    if os.path.isdir(data_fp):
        fps = [
            os.path.join(data_fp, fp)
            for fp in sorted(os.listdir(data_fp))
            if fp.endswith(".pt")
        ]
    else:
        fps = [data_fp]

    if not fps:
        raise RuntimeError("No .pt trajectories found in {}".format(data_fp))

    trajs = []
    for fp in fps:
        traj = torch.load(fp, map_location=device)
        trajs.append(traj)
    return trajs


def sample_batch(trajs, batch_size, horizon, obs_keys, device):
    samples = []
    for _ in range(batch_size):
        traj = random.choice(trajs)
        max_start = traj["action"].shape[0] - horizon
        if max_start <= 0:
            raise RuntimeError(
                "Trajectory is too short for horizon {}: {} steps".format(
                    horizon, traj["action"].shape[0]
                )
            )
        t = random.randrange(max_start)
        samples.append((traj, t))

    obs = {
        k: torch.stack([traj["observation"][k][t] for traj, t in samples]).to(device)
        for k in obs_keys + ["state"]
    }
    act = torch.stack(
        [traj["action"][t : t + horizon] for traj, t in samples]
    ).to(device)
    target_state = torch.stack(
        [traj["next_observation"]["state"][t : t + horizon] for traj, t in samples]
    ).to(device)

    return obs, act, target_state


def evaluate(model, trajs, horizon, obs_keys, device, n_samples=32):
    model.eval()
    errs = []
    with torch.no_grad():
        for _ in range(n_samples):
            obs, act, target_state = sample_batch(trajs, 1, horizon, obs_keys, device)
            pred = model.predict(obs, act, return_info=False, keys=["state"])["state"]
            err = (pred.mean[:, -1] - target_state[:, -1]).pow(2).sum(dim=-1).sqrt()
            errs.append(err.cpu())
    model.train()
    return torch.cat(errs).mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_fp", required=True)
    parser.add_argument("--train_data_fp", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    print("loading model:", args.model_fp)
    model = torch.load(args.model_fp, map_location=args.device)
    model = model.to(args.device)
    model.train()

    obs_keys = list(model.obs_keys)
    print("model obs_keys:", obs_keys)

    print("loading trajectories:", args.train_data_fp)
    trajs = load_trajs(args.train_data_fp, args.device)
    print("loaded {} trajectory file(s)".format(len(trajs)))
    print("trajectory lengths:", [traj["action"].shape[0] for traj in trajs])

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for itr in range(1, args.iters + 1):
        obs, act, target_state = sample_batch(
            trajs, args.batch_size, args.horizon, obs_keys, args.device
        )

        pred = model.predict(obs, act, return_info=False, keys=["state"])["state"]
        state_log_prob = pred.log_prob(target_state).sum(dim=-1)
        loss = -state_log_prob.mean()

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
        opt.step()

        if itr == 1 or itr % args.eval_every == 0:
            rmse = evaluate(model, trajs, args.horizon, obs_keys, args.device)
            print(
                "itr {:05d} | nll {:.6f} | final-step rmse {:.6f}".format(
                    itr, loss.item(), rmse
                ),
                flush=True,
            )

    out_fp = os.path.join(args.save_dir, "model.cpt")
    torch.save(model, out_fp)
    torch.save(model.input_normalizer, os.path.join(args.save_dir, "input_normalizer.cpt"))
    torch.save(model.output_normalizer, os.path.join(args.save_dir, "output_normalizer.cpt"))
    print("saved:", out_fp)


if __name__ == "__main__":
    main()
