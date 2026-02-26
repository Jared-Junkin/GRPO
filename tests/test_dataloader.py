import sys
sys.path.append("..")
from pathlib import Path
from utils import generate_dataset, load_dataset



def test_generate_dataset_round_trip_matches_file(tmp_path: Path) -> None:
    # parameters
    num_graphs = 25
    n = 20
    k = 20

    # temp file to write to
    writefile = tmp_path / "train_ds.txt"

    # generate DAG list and write them to a text file
    dags = generate_dataset(
        num_graphs=num_graphs,
        writefile=str(writefile),
        n=n,
        k=k,
    )

    loaded_dags = load_dataset(writefile=str(writefile))

    # quick sanity checks
    assert len(dags) == num_graphs
    assert len(loaded_dags) == num_graphs

    # line-by-line equality
    for i in range(len(loaded_dags)):
        assert dags[i] == loaded_dags[i], (
            f"{i} doesn't match!\n"
            f"dags[i][:100]={dags[i][:100]!r}\n"
            f"loaded_dags[i][:100]={loaded_dags[i][:100]!r}"
        )