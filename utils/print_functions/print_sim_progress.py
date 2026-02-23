import sys

def print_sim_progress(current_step, total_steps, bar_length=40):
    """
    Prints a clean block-style progress bar and percentage.
    Step counts are omitted to reduce printing overhead.
    """
    percent = float(current_step) / total_steps
    # Use the 'full block' character for a solid bar look
    filled_len = int(round(bar_length * percent))
    bar = '█' * filled_len + '░' * (bar_length - filled_len)

    # \r resets the line, end="" prevents a newline
    sys.stdout.write(f"\rPropagating: |{bar}| {percent:>4.0%}")
    sys.stdout.flush()

    if current_step == total_steps:
        print() # Move to the next line when finished