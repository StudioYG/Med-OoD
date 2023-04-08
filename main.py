from train import *
from toolkit import *
from train_ood import *
from hard_ood_construct import *

if __name__ == "__main__":  # Here
    train()
    hard_ood_construct()
    ood_ratios = np.linspace(0, 1, 11).round(2)
    dist_from_balance = []
    balance_approx = 0.65
    for i in ood_ratios:
        dist_from_balance.append(abs(estimate_ratios(ood_ratio=i) - balance_approx))
    best_idx = dist_from_balance.index(min(dist_from_balance))
    print('estimated ood ratios: ', ood_ratios[best_idx])
    train_ood(ood_ratio=ood_ratios[best_idx])
