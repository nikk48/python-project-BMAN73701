from __future__ import annotations

import os
from cli_utils import print_header, ask_int, ask_str, pause
from aed.analysis import run_aed_analysis
from aed.data_management import AEDDataManager, format_df_for_cli
from config import AUDIT_LOG_PATH
from pathlib import Path


def _aed_data_management_menu(file_path: str | Path) -> None:
    """Interactive CRUD-style management for the AED dataset."""

    mgr = AEDDataManager(file_path, id_col="ID", audit_log_path=AUDIT_LOG_PATH)
    info = mgr.load()

    while True:
        print_header("A&E DATA MANAGEMENT")
        print(f"Dataset: {info.file_path}")
        print(f"Rows: {len(mgr.df):,} | Columns: {mgr.df.shape[1]} | ID column: {info.id_col}")
        print(f"Unsaved changes: {'YES' if mgr.dirty else 'no'}")
        print("\nActions")
        print("1) Retrieve patient record by ID")
        print("2) Find patients within a numeric range (min/max) for a variable")
        print("3) Modify a patient field")
        print("4) Delete a patient record")
        print("5) Save changes")
        print("6) Save changes as... (new file)")
        print("7) Reload from disk (discard unsaved changes)")
        print("0) Back")

        choice = ask_int("Choose: ", default=1, valid={0, 1, 2, 3, 4, 5, 6, 7})
        if choice == 0:
            return

        if choice == 1:
            pid = ask_str("Enter patient ID (e.g., P10000): ")
            df = mgr.get_patient(pid)
            print("\n" + format_df_for_cli(df, max_rows=20))
            pause()

        elif choice == 2:
            col = ask_str("Column to filter (e.g., Age, LoS, noofpatients): ")
            raw_min = ask_str("Min value (blank for none): ", default="")
            raw_max = ask_str("Max value (blank for none): ", default="")
            min_v = float(raw_min) if raw_min.strip() != "" else None
            max_v = float(raw_max) if raw_max.strip() != "" else None
            try:
                df = mgr.filter_range(col, min_v, max_v)
                print(f"\nMatches: {len(df):,}")
                print(format_df_for_cli(df, max_rows=20))
            except Exception as e:
                print(f"\nCould not run range filter: {e}")
            pause()

        elif choice == 3:
            pid = ask_str("Patient ID to modify: ")
            col = ask_str("Column to modify: ")
            new_v = ask_str("New value: ")
            try:
                updated = mgr.update_patient_field(pid, col, new_v)
                if updated == 0:
                    print("No rows updated (ID not found).")
                else:
                    print(f"Updated {updated} row(s). New record preview:")
                    print(format_df_for_cli(mgr.get_patient(pid), max_rows=5))
            except Exception as e:
                print(f"Update failed: {e}")
            pause()

        elif choice == 4:
            pid = ask_str("Patient ID to delete: ")
            try:
                deleted, remaining = mgr.delete_patient(pid)
                if deleted == 0:
                    print("No rows deleted (ID not found).")
                else:
                    print(f"Deleted {deleted} row(s). Remaining rows: {remaining:,}")
            except Exception as e:
                print(f"Delete failed: {e}")
            pause()

        elif choice == 5:
            try:
                out = mgr.save()
                print(f"Saved to: {out}")
                print("(A timestamped backup was created next to the original CSV.)")
            except Exception as e:
                print(f"Save failed: {e}")
            pause()

        elif choice == 6:
            out_path = ask_str("Enter output CSV path: ")
            try:
                out = mgr.save(out_path=out_path)
                print(f"Saved to: {out}")
            except Exception as e:
                print(f"Save-as failed: {e}")
            pause()

        elif choice == 7:
            try:
                info = mgr.load()
                print("Reloaded from disk.")
            except Exception as e:
                print(f"Reload failed: {e}")
            pause()


def aed_menu(file_path, seed: int, n: int) -> None:
    while True:
        print_header("A&E MODULE")
        print("1) Run A&E sample + summary + charts (Task 4) + drivers (Task 5)")
        print("2) Interactive data management (lookup / range filter / modify / delete)")
        print("0) Back")

        choice = ask_int("Choose: ", default=1, valid={0, 1, 2})
        if choice == 0:
            return
        if choice == 1:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"AED CSV not found: {file_path}")
            run_aed_analysis(file_path=file_path, seed=seed, n=n)
            pause()
        elif choice == 2:
            _aed_data_management_menu(file_path)
