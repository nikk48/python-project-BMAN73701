# BMAN73701 Python Project

## What’s inside

1. **LP scheduling** (Tasks 1–3)
2. **A&E descriptive analysis + drivers** (Tasks 4–5)
3. **A&E breach prediction (ML)** (Task 6)
4. **A&E interactive data management (extension)**
   - Look up a patient by ID
   - Filter patients by a numeric range for any variable
   - Modify a patient field (edit a value)
   - Delete a patient record
   - Save changes (creates an automatic timestamped backup)
   - Audit log of user actions

## How to run

From the project folder:

```bash
python main.py
```

Then choose **A&E module → Interactive data management**.

## Audit log

User actions (lookups, filters, edits, deletes, saves) are written to an audit log.

Default location:

- `./logs/audit.log`

If the program cannot write to that directory (permissions), it automatically falls back to a temporary directory on your machine.
