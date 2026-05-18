import subprocess
import pandas as pd
import matplotlib.pyplot as plt


def load_grf(start_frame=None, end_frame=None) -> pd.DataFrame:
    path = f"/mnt/g/Shared drives/NMBL Shared Data/datasets/{dset_name}/{dset_name}_Formatted_No_Arm/{subject_name}/trials/{trial_name}/grf.mot"
    # .mot files have a variable number of header rows; skip until the numeric data block
    with open(path, "r") as f:
        lines = f.readlines()

    # Find the line index that contains the column headers (starts with 'time' or similar)
    header_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and not stripped[0].isdigit() and not stripped.startswith("-"):
            # Keep updating until we hit the last non-numeric header line
            header_idx = i

    nrows = None
    if start_frame is not None or end_frame is not None:
        sf = start_frame or 0
        if end_frame is not None:
            nrows = end_frame - sf

    skiprows_val = header_idx if start_frame is None else list(range(header_idx)) + list(range(header_idx + 1, header_idx + 1 + (start_frame or 0)))

    df = pd.read_csv(path, sep="\t", skiprows=skiprows_val, nrows=nrows, engine="python")
    return df


def main(start_frame=None, end_frame=None):
    # mount drive if cannot find the data

    df = load_grf(start_frame=start_frame, end_frame=end_frame)

    total_vy = df["R_ground_force_vy"] + df["L_ground_force_vy"]
    median_val = total_vy.median()
    weight_kg = median_val / 9.81

    print(f"Median total vertical GRF : {median_val:.4f} N")
    print(f"Estimated body weight     : {weight_kg:.4f} kg")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(total_vy.values, label="R + L vertical GRF")
    ax.axhline(median_val, color="red", linestyle="--", linewidth=1.5,
               label=f"Median = {median_val:.2f} N  ({weight_kg:.2f} kg)")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Vertical GRF (N)")
    ax.set_title(f"Total Vertical Ground Reaction Force\n{dset_name} / {trial_name}")
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    dset_name = "Moore2015"
    subject_name = "subject17"
    trial_name = "T075"
    start_frame = 1600 
    end_frame = 2400

    main(start_frame=start_frame, end_frame=end_frame)
