import albumentations as A


transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.75, rotate_limit=180, scale_limit=0.2, shift_limit=0.1),
        A.GaussNoise(p=0.5, var_limit=5),
    ]
)
