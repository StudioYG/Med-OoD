# References:
# https://github.com/qubvel/segmentation_models.pytorch
# https://pyimagesearch.com/2018/09/03/semantic-segmentation-with-opencv-and-deep-learning/

from segmentation_models_pytorch import utils
from toolkit import *
from tqdm import tqdm
from config import *


def hard_ood_construct():
    for str_code in [79, 158, 237]:
        setup_seed(str_code)
        df = pd.read_csv(os.path.join('k_folds', str(str_code) + 'Image_Patchs.csv'))

        train_ids, test_ids = read_csv(df)

        images_dir, masks_dir = 'OoD_Patchs', 'Mask_Patchs'
        net, ENCODER, ENCODER_WEIGHTS, CLASSES, ACTIVATION, model = get_config()
        DEVICE = 'cuda'

        preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

        ood_dataset = OoD_Dataset(images_dir, masks_dir, train_ids, classes=CLASSES,
                                  preprocessing=get_preprocessing(preprocessing_fn))

        loss = utils.losses.DiceLoss()

        metrics = [
            utils.metrics.IoU(threshold=0.5),
        ]

        # load best saved checkpoint
        best_model = torch.load(os.path.join('models', str(str_code) + '_' + net + '_' + ENCODER + '.pth'))

        # summary(best_model, torch.zeros((1, 3, 128, 128)).to(DEVICE)).to_string('model_summary.txt', index=True)

        ood_dataloader = DataLoader(ood_dataset)

        # evaluate model on test set
        test_epoch = smp.utils.train.ValidEpoch(
            model=best_model,
            loss=loss,
            metrics=metrics,
            device=DEVICE,
        )

        hard_ood = []
        for i, (input, gt) in tqdm(enumerate(ood_dataloader)):
            pr = best_model(input.to(DEVICE))
            iou_score = metrics[0].forward(pr, gt.to(DEVICE))
            if iou_score < 1.0:
                hard_ood.append(train_ids[i])
        df = pd.DataFrame({'hard_ood': np.array(hard_ood)})
        df.to_csv(os.path.join('HardOoD', str(str_code) + '_' + net + '_' + ENCODER + '_' + "HardOoD.csv"))
 


if __name__ == "__main__":  # Here
    hard_ood_construct()
