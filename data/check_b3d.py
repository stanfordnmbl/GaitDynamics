import nimblephysics as nimble


b3d_file = '/mnt/d/Downloads/AB06_split0.b3d'
subject = nimble.biomechanics.SubjectOnDisk(b3d_file)
for trial_id in range(subject.getNumTrials()):
    trial_name = subject.getTrialName(trial_id)
    print(trial_name)

    missing_grf_labels = subject.getMissingGRF(trial_id)

    probably_missing = [reason != nimble.biomechanics.MissingGRFReason.notMissingGRF for reason
                        in subject.getMissingGRF(trial_id)]










