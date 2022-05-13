"""
Assume <blk> = 0 in tokenizer.
"""
import sys

if __name__ == "__main__":
    if len(sys.argv[1:]) == 0:
        sys.stderr.write("vocab size is required.\n")
        sys.exit(1)

    vocab_size = int(sys.argv[1])
    assert vocab_size >= 1
    # state:0 -> state:0, input: <blk>, output: <eps>
    sys.stdout.write("0 0 1 0\n")
    sys.stdout.write("0\n")
    for i in range(1, vocab_size):
        sys.stdout.write(f"0 {i} {i+1} {i}\n")
        sys.stdout.write(f"{i} {i} {i+1} 0\n")
        sys.stdout.write(f"{i} 0 {i+1} 0\n")

    for prev_s in range(1, vocab_size):
        for next_s in range(1, vocab_size):
            if prev_s != next_s:
                sys.stdout.write(f"{prev_s} {next_s} {next_s+1} {next_s}\n")

        sys.stdout.write(f"{prev_s}\n")
