import matplotlib.pyplot as plt
from extract_best import root, results_path, isfile, pd, os, dirs


def main():
    for d in dirs():
        # Load results
        df = pd.read_csv(results_path(d))

        # Plot results
        fig, ax = plt.subplots(layout="constrained")
        df.plot(y="y", ax=ax)
        ax.grid()
        ax.set_xlabel("Iterations", fontsize=16)
        ax.set_ylabel("y", fontsize=16)
        path = os.path.join(root, d, "traj.png")
        fig.savefig(path)


if __name__ == "__main__":
    main()
