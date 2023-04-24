import albumentations as A


transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.75),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
        A.CoarseDropout(
            max_holes=1,
            max_width=0.3,
            max_height=0.3,
            mask_fill_value=0,
            p=0.5,
        ),
        A.OneOf(
            [
                A.GaussianBlur(),
                A.MotionBlur(),
            ],
            p=0.4,
        ),
    ]
)
