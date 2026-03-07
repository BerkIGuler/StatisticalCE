from src.data.dataset import TDLDataset, get_in_distribution_test_datasets
from src.models.bilinear_interp import BilinearInterpolation
from argparse import ArgumentParser
import numpy as np
from pathlib import Path
from tqdm import tqdm
import yaml


def main():
    parser = ArgumentParser()
    parser.add_argument("--delay_profile", type=str, default="A")
    parser.add_argument("--eval_SNRs", nargs="+", type=int, default=[0, 5, 10, 15, 20, 25, 30])
    parser.add_argument("--pilot_symbols", nargs="+", type=int, default=[2])
    parser.add_argument("--save_dir", type=str, default="results")
    args = parser.parse_args()

    print(f"Algorithm: bilinear_interp")
    print(f"Delay profile: TDL-{args.delay_profile}")
    print(f"Eval SNRs: {args.eval_SNRs}")
    print(f"Pilot symbols: {args.pilot_symbols}")
    print(f"Save dir: {args.save_dir}")

    test_path = f"/opt/shared/datasets/NeoRadiumTDLdataset/test/TDL{args.delay_profile}"
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    model = BilinearInterpolation()

    print(f"\nStarting evaluation on test data from {test_path}")
    results = {}
    for SNR in args.eval_SNRs:
        results[SNR] = {}
        test_datasets_per_snr = get_in_distribution_test_datasets(
            test_path,
            return_pilots_only=False,
            SNRs=[SNR],
            pilot_symbols=args.pilot_symbols,
        )
        for name, test_dataset in test_datasets_per_snr:
            results[SNR][name] = {}
            nmses = []
            for data in tqdm(test_dataset, desc=f"Evaluating {name} at SNR {SNR}"):
                hp_ls, h_true, stats = data
                hp_ls = hp_ls.numpy()
                h_true = h_true.numpy()

                assert SNR == stats["SNR"], f"SNR {SNR} does not match {stats['SNR']}"

                h_true_hat = model(hp_ls)

                mse = np.mean(np.abs(h_true_hat - h_true) ** 2)
                h_true_power = np.mean(np.abs(h_true) ** 2)
                nmse = mse / h_true_power
                nmses.append(nmse)
            results[SNR][name]["nmse_mean_linear"] = float(np.mean(nmses))
            results[SNR][name]["nmse_mean_db"] = float(10 * np.log10(np.mean(nmses)))

    pilot_symbols_str = "".join(map(str, args.pilot_symbols))
    save_path = f"{args.save_dir}/bilinear_interp_{args.delay_profile}_{pilot_symbols_str}.yaml"
    with open(save_path, "w") as f:
        yaml.dump(results, f)
    print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()
