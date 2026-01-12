from __future__ import annotations

from cli_utils import print_header, ask_int, pause
from config import AED_FILE_PATH, AED_RANDOM_SEED, AED_SAMPLE_N, TASK6_TARGET_RECALL

from lp.ui import lp_menu
from aed.ui import aed_menu
from ml.ui import ml_menu
import pandas as pd

def main() -> None:
    while True:
        print_header("MAIN MENU")
        print("1) LP scheduling tasks (Task 1/2/3)")
        print("2) A&E sample + summary + charts (Task 4) + drivers (Task 5)")
        print("3) A&E breach prediction (Task 6 — ML)")
        print("0) Quit")

        choice = ask_int("Choose: ", default=1, valid={0, 1, 2, 3})
        if choice == 0:
            print("Goodbye.")
            return

        if choice == 1:
            lp_menu()
        elif choice == 2:
            aed_menu(file_path=AED_FILE_PATH, seed=AED_RANDOM_SEED, n=AED_SAMPLE_N)
        elif choice == 3:
            ml_menu(file_path=AED_FILE_PATH, seed=AED_RANDOM_SEED, target_recall=TASK6_TARGET_RECALL)

        pause()

if __name__ == "__main__":
    main()
