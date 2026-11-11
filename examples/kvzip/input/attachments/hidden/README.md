# Judge-only attachments (not published)

The expert's `needles.json` lives here in a real input folder: three hidden needles (needle,
question, answer) that only the Judge may see. They are not published with this example, because
publishing them would reveal hidden test cases.

To rebuild the task, put your own `needles.json` here, a JSON list of
`{"needle": "...", "question": "...", "answer": "..."}` objects.
To use them with the delivered task in `examples/kvzip/task/`, copy the file to the task's
`judge/private/needles.json`; its hidden suite's hash is frozen in `judge/suite_manifest.json`, so
then run the task's `scripts/build_suites.sh --refreeze` once to record the new suites.
