import torch


def fit(batches: tp.List[tp.List], loader, preprocessor, model, optimizer, loss_fn, 
    epochs: int, ckpt_path: tp.Optional[str]=None, ckpt_per_iter: tp.Optional[int]=None,
    verbose_per_iter: tp.Optional[int]=None, history_path: tp.Optional[str]=None,
    meta_path: tp.Optional[str]=None, heatmap_input: tp.Optional[str]=None,
    heatmap_output: tp.Optional[str]=None, heatmap_per_iter: tp.Optional[int]=None):

    n_batches = len(batches)

    # GPU is preferred
    device = get_device()

    # Load model from checkpoint
    if is_loadable(ckpt_path):
        print("~ ~ ~ ~ ~ ~ ~ ~ ~ ~")
        print("Checkpoint was found")
        print("~ ~ ~ ~ ~ ~ ~ ~ ~ ~")
        print()
        checkpoint = load_checkpoint(ckpt_path, device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        loss_fn.load_state_dict(checkpoint['loss_fn'])

    # Load history
    if is_loadable(history_path) and history_path is not None:
        history = load_json(history_path)
    elif not is_loadable(history_path) and history_path is not None:
        history = create_history()

    # Load meta
    if is_loadable(meta_path) and meta_path is not None:
        meta = load_json(meta_path)
    elif not is_loadable(meta_path) and meta_path is not None:
        meta = create_meta()

    # Read parameters
    start_epoch = meta["epoch"]
    start_batch = meta["batch"]

    # Switch to device
    model.to(device)
    model.train()
    optimizer_to(optimizer, device)

    for epoch in range(start_epoch, epochs):

        subset_loss = []
        subset_miou = []
        subset_mpa = []

        for i in range(start_batch, len(batches)):

            batch = batches[i]

            # Data
            torch.cuda.empty_cache()
            loaded = loader(batch)
            x, y = preprocessor(loaded[:])
            x = x.to(device).float()
            y = y.to(device).float()

            # Step
            pred = model(x)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Metrics
            miou = get_mean_iou(pred, y)
            mpa = get_mean_pixel_acc(pred, y)

            # Add to buffer
            subset_loss.append(loss.item())
            subset_miou.append(miou)
            subset_mpa.append(mpa)

            # Add to history
            history['Loss'].append(loss.item())
            history['PixelAccuracy'].append(mpa)
            history['mIOU'].append(miou)

            # change meta batch
            meta["batch"] += 1

            # Verbose state of training
            if verbose_per_iter is not None:
                if (i+1) % verbose_per_iter == 0:
                    show_state(epoch, i+1, subset_loss, subset_miou, subset_mpa)
                    subset_loss = []
                    subset_miou = []
                    subset_mpa = []

            # Create checkpoint
            if (ckpt_path is not None) and (ckpt_per_iter is not None):
                if (i+1) % ckpt_per_iter == 0:
                    if meta_path is not None:
                        save_json(meta_path, meta)
                    if history_path is not None:
                        save_json(history_path, history)
                    create_checkpoint(ckpt_path, model, optimizer, loss_fn)
            
            if (heatmap_input is not None) and (heatmap_output is not None) and (heatmap_per_iter is not None):
                if (i+1) % heatmap_per_iter == 0:
                    if heatmap_output[-1] == "/":
                        current_output = f"{heatmap_output}{epoch}{i+1}"
                    else:
                        current_output = f"{heatmap_output}/{epoch}{i+1}"
                    create_folder(current_output)
                    create_heatmap_from_folder(heatmap_input, current_output,
                        model, preprocessor, device, alpha=0.4)
            
        # change meta epoch
        meta["epoch"] += 1
        meta["batch"] = 0
        start_batch = 0
