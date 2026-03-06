from src.data.dataset import TDLDataset
from src.models.lmmse import LMMSE

dataset = TDLDataset(data_path="data/tdl", file_size=1000, normalization_stats=None, return_pilots_only=True, num_subcarriers=120, num_symbols=14, SNRs=[0, 5, 10, 15, 20, 25, 30], pilot_symbols=[2, 11], pilot_every_n=2)
lmmse = LMMSE(num_pilot_subcarriers=120, num_pilot_symbols=2, num_subcarriers=120, num_symbols=14)
lmmse.fit(dataset)

for batch in dataset:
    hp_ls, h_true, _ = batch
    h_true_hat = lmmse(hp_ls, 1)
    print(h_true_hat)
    break


def main():
    delay_profile = "A"
    train_path = f"/opt/shared/datasets/NeoRadiumTDLdataset/train/TDL{delay_profile}"
    test_path = f"/opt/shared/datasets/NeoRadiumTDLdataset/test/TDL{delay_profile}"

if __name__ == "__main__":
    main()