import argparse
import re
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="*", type=str)
    parser.add_argument("--output", "-o", type=str, required=True)
    parser.add_argument("--data", "-d", type=str, default="./data")
    parser.add_argument("--split", "-s", type=str, default="test")
    parser.add_argument("--num", "-n", type=int, default=500)

    args = parser.parse_args()

    stats = {}
    files = args.inputs
    print(files)
    for filename in files:
        with open(filename, "r") as f:
            content = str(f.readlines())
            pat = r"SENTENCE\s+(?P<id>[0-9]+)(.|\n)+?Errors\s+=\s+(?P<wer>[0-9\.]+)"
            # pat = r"SENTENCE\s+(?P<id>[0-9]+)(.|\n)+?Errors"
            for m in re.finditer(pat, content):
                sid = int(m.group("id"))
                wer = float(m.group("wer"))

                if sid not in stats:
                    stats[sid] = []
                stats[sid] += [wer]

    for sid in stats:
        stats[sid] = sum(stats[sid])

    selection = sorted(stats.items(), key=lambda x: x[1])[:args.num]
    selection = [x[0] for x in selection]

    with open("test.best_id", "w") as f:
        for s in selection:
            f.write(f"{s}\n")

    data_list = {}
    prefix = Path(args.data) / args.split
    for suffix in [".wav_list", ".en", ".zh-CN"]:
        data_list[suffix] = []
        with open(prefix.as_posix() + suffix, "r") as f:
            for line in f:
                data_list[suffix] += [line.strip()]

    prefix = Path(args.output) / "test"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix, dlist in data_list.items():
        filtered = [dlist[i - 1] for i in selection]
        with open(prefix.as_posix() + suffix, "w") as f:
            for line in filtered:
                f.write(f"{line}\n")
