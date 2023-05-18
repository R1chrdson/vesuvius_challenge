import albumentations as A


transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(p=1, rotate_limit=180, scale_limit=0.3, shift_limit=0.3),
        A.GaussNoise(p=0.5, var_limit=5),
    ]
)
