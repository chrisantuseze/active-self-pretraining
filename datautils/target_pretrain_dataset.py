from datautils import dataset_enum, ucmerced, sketch, clipart


def get_target_pretrain_ds(args):
    if args.target_dataset == dataset_enum.DatasetType.UCMERCED.value:
        return ucmerced.UCMerced(args)
    
    elif args.target_dataset == dataset_enum.DatasetType.SKETCH.value:
        return sketch.Sketch(args)

    elif args.target_dataset == dataset_enum.DatasetType.CLIPART.value:
        return clipart.Clipart(args)

    else:
        NotImplementedError