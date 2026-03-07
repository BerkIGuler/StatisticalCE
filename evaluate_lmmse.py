from src.data.dataset import TDLDataset, get_in_distribution_test_datasets
from src.models.lmmse import LMMSE
from argparse import ArgumentParser
import yaml
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm


def main():
    parser = ArgumentParser()
    parser.add_argument("--delay_profile", type=str, default="A")
    parser.add_argument("--eval_SNRs", nargs="+", type=int, default=[0, 5, 10, 15, 20, 25, 30])
    parser.add_argument("--pilot_symbols", nargs="+", type=int, default=[2])
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--lmmse_stats", type=str, default="train", choices=["train", "per_test"],
                        help="'train': fit LMMSE on training data (averaged stats). "
                             "'per_test': fit LMMSE per test scenario (matched stats).")
    args = parser.parse_args()

    print(f"Algorithm: lmmse")
    print(f"Delay profile: TDL-{args.delay_profile}")
    print(f"Eval SNRs: {args.eval_SNRs}")
    print(f"Pilot symbols: {args.pilot_symbols}")
    print(f"Save dir: {args.save_dir}")
    print(f"LMMSE stats: {args.lmmse_stats}")

    train_path = f"/opt/shared/datasets/NeoRadiumTDLdataset/train/TDL{args.delay_profile}"
    test_path = f"/opt/shared/datasets/NeoRadiumTDLdataset/test/TDL{args.delay_profile}"
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    if args.lmmse_stats == "train":
        with open(f"{train_path}/metadata.yaml", "r") as f:
            train_metadata = yaml.safe_load(f)
        train_file_size = train_metadata["config"]["num_channels_per_config"]

        print(f"Loading training data from {train_path}...")
        train_dataset = TDLDataset(
            data_path=train_path,
            file_size=train_file_size,
            return_pilots_only=True,
            SNRs=[100],
            pilot_symbols=args.pilot_symbols,
        )
        print(f"Training dataset loaded: {len(train_dataset)} samples")
        train_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=False)
        model = LMMSE(
            num_pilot_subcarriers=train_dataset.num_pilot_subcarriers,
            num_pilot_symbols=train_dataset.num_pilot_symbols,
            num_subcarriers=train_dataset.num_subcarriers,
            num_symbols=train_dataset.num_symbols,
        )
        print("Fitting LMMSE statistics on training data...")
        model.fit(train_dataloader)
        print("LMMSE fitting complete.")

    print(f"\nStarting evaluation on test data from {test_path}")
    results = {}
    for SNR in args.eval_SNRs:
        results[SNR] = {}
        test_datasets_per_snr = get_in_distribution_test_datasets(
            test_path,
            return_pilots_only=True,
            SNRs=[SNR],
            pilot_symbols=args.pilot_symbols,
        )
        for name, test_dataset in test_datasets_per_snr:
            if args.lmmse_stats == "per_test":
                fit_dataset = TDLDataset(
                    data_path=Path(test_path) / name,
                    file_size=test_dataset.file_size,
                    return_pilots_only=True,
                    SNRs=[100],
                    pilot_symbols=args.pilot_symbols,
                )
                fit_dataloader = DataLoader(fit_dataset, batch_size=512, shuffle=False)
                model = LMMSE(
                    num_pilot_subcarriers=fit_dataset.num_pilot_subcarriers,
                    num_pilot_symbols=fit_dataset.num_pilot_symbols,
                    num_subcarriers=fit_dataset.num_subcarriers,
                    num_symbols=fit_dataset.num_symbols,
                )
                print(f"Fitting LMMSE on {name} ({len(fit_dataset)} samples)...")
                model.fit(fit_dataloader)

            results[SNR][name] = {}
            nmses = []
            for data in tqdm(test_dataset, desc=f"Evaluating {name} at SNR {SNR}"):
                hp_ls, h_true, stats = data
                hp_ls = hp_ls.numpy()
                h_true = h_true.numpy()

                assert SNR == stats["SNR"], f"SNR {SNR} does not match {stats['SNR']}"

                hp_ls = hp_ls.flatten()
                h_true_hat = model(hp_ls, test_dataset.noise_variance)
                h_true_hat = h_true_hat.reshape(test_dataset.num_subcarriers, test_dataset.num_symbols)

                mse = np.mean(np.abs(h_true_hat - h_true) ** 2)
                h_true_power = np.mean(np.abs(h_true) ** 2)
                nmse = mse / h_true_power
                nmses.append(nmse)
            results[SNR][name]["nmse_mean_linear"] = float(np.mean(nmses))
            results[SNR][name]["nmse_mean_db"] = float(10 * np.log10(np.mean(nmses)))

    pilot_symbols_str = "".join(map(str, args.pilot_symbols))
    save_path = f"{args.save_dir}/lmmse_{args.delay_profile}_{args.lmmse_stats}_{pilot_symbols_str}.yaml"
    with open(save_path, "w") as f:
        yaml.dump(results, f)
    print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()
