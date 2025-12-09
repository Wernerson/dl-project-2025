from frechet_music_distance import FrechetMusicDistance

def calculate_FMD(gt_dataset, pred_dataset, log_wandb = False):
    fmd = FrechetMusicDistance(feature_extractor='clamp2', gaussian_estimator='shrinkage', verbose=True)
    # TODO implement FMD-inf with small generated sets?

    score = fmd.score(
        reference_path="datasets/real_midi",
        test_path="datasets/generated_midi"
    )

    print("FMD score: ", score)

    if log_wandb:
        # TODO weights and biases logging
        pass


# TODO embedding of generated sequence