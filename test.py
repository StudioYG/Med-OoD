# References:
# https://github.com/qubvel/segmentation_models.pytorch
# https://pyimagesearch.com/2018/09/03/semantic-segmentation-with-opencv-and-deep-learning/

from segmentation_models_pytorch import utils
from toolkit import *


def main():
    train_evals = []
    test_evals = []
    vis = False
    for str_code in [79, 158, 237]:
        setup_seed(str_code)
        df = pd.read_csv(os.path.join('k_folds', str(str_code) + 'Image_Patchs.csv'))

        train_ids, test_ids = read_csv(df)

        images_dir, masks_dir = 'Image_Patchs', 'Mask_Patchs'
        net = 'unet'
        ENCODER = 'vgg11_bn'
        ENCODER_WEIGHTS = 'imagenet'
        CLASSES = ['0', '1', '2', '3', '4', '5', '6']  # 7 classes: background + 6 classes
        DEVICE = 'cuda'

        # load best saved checkpoint
        # best_model = torch.load(os.path.join('models', str(str_code) + '_' + net + '_' + ENCODER + '_ood.pth'))
        best_model = torch.load(os.path.join('models', str(str_code) + '_' + net + '_' + ENCODER + '.pth'))

        preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

        train_dataset = Dataset(images_dir, masks_dir, train_ids, classes=CLASSES,
                                preprocessing=get_preprocessing(preprocessing_fn))
        test_dataset = Dataset(images_dir, masks_dir, test_ids, classes=CLASSES,
                               preprocessing=get_preprocessing(preprocessing_fn))
        loss = utils.losses.DiceLoss()

        metrics = [
            utils.metrics.IoU(threshold=0.5),
        ]



        summary(best_model, torch.zeros((1, 3, 128, 128)).to(DEVICE)).to_string('model_summary.txt', index=True)

        train_dataloader = DataLoader(train_dataset)
        test_dataloader = DataLoader(test_dataset)

        # evaluate model on test set
        test_epoch = smp.utils.train.ValidEpoch(
            model=best_model,
            loss=loss,
            metrics=metrics,
            device=DEVICE,
        )

        train_logs = test_epoch.run(train_dataloader)
        test_logs = test_epoch.run(test_dataloader)
        train_evals.append(train_logs['iou_score'])
        test_evals.append(test_logs['iou_score'])

        if vis:
            # test dataset without transformations for image visualization
            test_dataset_vis = Dataset(images_dir, masks_dir, test_ids, classes=CLASSES, )

            for i in range(5):
                n = np.random.choice(len(test_dataset))

                image_vis = test_dataset_vis[n][0].astype('uint8')
                image, gt_mask = test_dataset[n]
                gt_masks = [(gt_mask[c, ...] * c) for c in range(gt_mask.shape[0])]
                gt_mask = np.sum(gt_masks, axis=0).astype('uint8')

                x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
                pr_mask = best_model.predict(x_tensor)
                pr_mask = (torch.argmax(pr_mask.squeeze(), dim=0).cpu().numpy().astype('uint8'))

                color_map = np.load('color_map.npy')
                gt_mask = color_map[gt_mask]
                pr_mask = color_map[pr_mask]

                legend = np.zeros(((len(CLASSES) * 25) + 25, 300, 3), dtype="uint8") + 255

                CLASSES = ['Background', 'Neutrophil', 'Epithelial', 'Lymphocyte', 'Plasma', 'Neutrophil',
                           'Connective tissue']

                for (i, (cls, color)) in enumerate(zip(CLASSES, color_map)):
                    # draw the class name + color on the legend
                    color = [int(c) for c in color]
                    cv2.putText(legend, cls, (5, (i * 25) + 17),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    cv2.rectangle(legend, (150, (i * 25)), (300, (i * 25) + 25),
                                  tuple(color), -1)

                visualize(
                    image=image_vis,
                    ground_truth_mask=gt_mask,
                    predicted_mask=pr_mask,
                    legend=legend,
                )
    print('IoU evaluations on training set:', train_evals)
    print('IoU average performance on training set:', str(np.mean(train_evals)) + '+-' + str(np.std(train_evals)))
    print('IoU evaluations on testing set:', test_evals)
    print('IoU average performance on testing set:', str(np.mean(test_evals)) + '+-' + str(np.std(test_evals)))


if __name__ == "__main__":  # Here
    main()
