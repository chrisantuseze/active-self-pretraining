from datautils import dataset_enum, ucmerced, sketch, clipart


def get_target_pretrain_ds(args, isAL=True):
    if args.target_dataset == dataset_enum.DatasetType.UCMERCED.value:
        print("using the UCMERCED dataset")
        return ucmerced.UCMerced(args, isAL)
    
    elif args.target_dataset == dataset_enum.DatasetType.SKETCH.value:
        print("using the SKETCH dataset")
        return sketch.Sketch(args, isAL)

    elif args.target_dataset == dataset_enum.DatasetType.CLIPART.value:
        print("using the CLIPART dataset")
        return clipart.Clipart(args, isAL)

    else:
        NotImplementedError