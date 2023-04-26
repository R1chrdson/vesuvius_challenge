import albumentations as A


transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
        A.ShiftScaleRotate(p=0.75),
    ]
)
