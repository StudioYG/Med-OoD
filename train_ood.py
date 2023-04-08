# References:
# https://github.com/qubvel/segmentation_models.pytorch/blob/master/examples/cars%20segmentation%20(camvid).ipynb

from toolkit import *
from config import *
from torch.utils.data import ConcatDataset
from segmentation_models_pytorch import utils


def train_ood(ood_ratio):
    iou_evals = []
    for str_code in [79, 158, 237]:
        setup_seed(str_code)
        df = pd.read_csv(os.path.join('k_folds', str(str_code) + 'Image_Patchs.csv'))

        train_ids, test_ids = read_csv(df)

        images_dir, masks_dir = 'Image_Patchs', 'Mask_Patchs'
        images_ood_dir, masks_ood_dir = 'OoD_Patchs', 'Mask_Patchs'
        net, ENCODER, ENCODER_WEIGHTS, CLASSES, ACTIVATION, model = get_config()
        DEVICE = 'cuda'
        num_epochs = 36

        df_ood = pd.read_csv(
            os.path.join('HardOoD', str(str_code) + '_' + net + '_' + ENCODER + '_' + "HardOoD.csv"))
        ood_ids = []
        for i, row in df_ood.iterrows():
            ood_ids.append(row['hard_ood'])

        random.shuffle(ood_ids)
        ood_ids = ood_ids[:round(len(ood_ids) * ood_ratio)]

        preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

        train_dataset = Dataset(images_dir, masks_dir, train_ids, classes=CLASSES,
                                augmentation=get_train_augmentation(),
                                preprocessing=get_preprocessing(preprocessing_fn))
        ood_dataset = OoD_Dataset(images_ood_dir, masks_ood_dir, ood_ids, classes=CLASSES,
                                  preprocessing=get_preprocessing(preprocessing_fn))

        combine_dataset = ConcatDataset([train_dataset, ood_dataset])
        train_loader = DataLoader(combine_dataset, batch_size=8, shuffle=True, num_workers=12)
        test_dataset = Dataset(images_dir, masks_dir, test_ids, classes=CLASSES,
                               preprocessing=get_preprocessing(preprocessing_fn))
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=6)

        # summary(model, torch.zeros((1, 3, 128, 128))).to_string('model_summary.txt', index=True)

        loss = utils.losses.DiceLoss()

        metrics = [
            utils.metrics.IoU(threshold=0.5),
        ]

        optimizer = torch.optim.Adam([
            dict(params=model.parameters(), lr=0.0001),
        ])

        # it is a simple loop of iterating over dataloader`s samples
        train_epoch = smp.utils.train.TrainEpoch(
            model,
            loss=loss,
            metrics=metrics,
            optimizer=optimizer,
            device=DEVICE,
            verbose=True,
        )

        valid_epoch = smp.utils.train.ValidEpoch(
            model,
            loss=loss,
            metrics=metrics,
            device=DEVICE,
            verbose=True,
        )

        max_score = 0

        x_epoch_data = []
        train_dice_loss = []
        train_acc_score = []
        valid_dice_loss = []
        valid_acc_score = []

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()

        # train model for 36 epochs
        for i in range(0, num_epochs):

            print('\nEpoch: {}'.format(i))
            train_logs = train_epoch.run(train_loader)
            valid_logs = valid_epoch.run(test_loader)

            x_epoch_data.append(i)
            train_dice_loss.append(train_logs['dice_loss'])
            train_acc_score.append(train_logs['iou_score'])
            valid_dice_loss.append(valid_logs['dice_loss'])
            valid_acc_score.append(valid_logs['iou_score'])

            # do something (save model, change lr, etc.)
            if max_score < valid_logs['iou_score']:
                max_score = valid_logs['iou_score']
                torch.save(model, './best_model.pth')
                print('Model saved!')

            if i == 25:
                optimizer.param_groups[0]['lr'] = 1e-5
                print('Decrease decoder learning rate to 1e-5!')

        end.record()

        # Waits for everything to finish running
        torch.cuda.synchronize()

        print("[INFO] Time elapsed: " + str(start.elapsed_time(end) / 60000) + " minutes")  # milliseconds

        fig = plt.figure(figsize=(14, 5))

        ax1 = fig.add_subplot(1, 2, 1)
        line1, = ax1.plot(x_epoch_data, train_dice_loss, label='train')
        line2, = ax1.plot(x_epoch_data, valid_dice_loss, label='validation')
        ax1.set_title("dice loss")
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('dice_loss')
        ax1.legend(loc='upper right')

        ax2 = fig.add_subplot(1, 2, 2)
        line1, = ax2.plot(x_epoch_data, train_acc_score, label='train')
        line2, = ax2.plot(x_epoch_data, valid_acc_score, label='validation')
        ax2.set_title("accuracy")
        ax2.set_xlabel('epoch')
        ax2.set_ylabel('accuracy')
        ax2.legend(loc='upper left')
        plt.savefig(os.path.join('models', str(str_code) + '_' + net + '_' + ENCODER + '_ood_trainlogs' + '.png'))
        # plt.show()

        # load best saved checkpoint
        best_model = torch.load('./best_model.pth')
        torch.save(best_model, os.path.join('models', str(str_code) + '_' + net + '_' + ENCODER + '_ood.pth'))

        test_dataloader = DataLoader(test_dataset)

        # evaluate model on test set
        test_epoch = smp.utils.train.ValidEpoch(
            model=best_model,
            loss=loss,
            metrics=metrics,
            device=DEVICE,
        )

        logs = test_epoch.run(test_dataloader)
        iou_evals.append(logs['iou_score'])
    print('IoU evaluations on three repeated experiments:', iou_evals)
    print('IoU average performance:', str(np.mean(iou_evals)) + '+-' + str(np.std(iou_evals)))


if __name__ == "__main__":  # Here
    train_ood(ood_ratio=0.55)
