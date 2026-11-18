# Source review scope

The cleanup removes site-specific deployment tooling, authentication
configuration, service endpoints, storage identifiers, personal paths and fixed
organization references from the current source files. Public data acquisition
URLs are limited to cited sources; local model locations are supplied by the
operator. Saved notebook outputs and
execution metadata are cleared to avoid distributing session and dataset dumps.
Third-party licenses and research citations are retained.

The public PRES dataset URL and its quoted repository ID are explicitly allowed
as dataset provenance, as requested for the preparation instructions. Other
organization references and internal service endpoints remain audit findings.

The audit scans the current versioned files plus non-ignored additions. It checks
known unwanted references, credential formats, private endpoint patterns, Python
syntax, notebook structure and saved notebook outputs. It reports filenames,
line numbers and rule names without printing matched credentials.

This is a static source review, not a proof that every possible confidential
detail is absent. Generated figures and table snapshots were removed during source minimization. The full GPU experiments require data and their training environments;
the small offline tests do not validate model accuracy or reproduce paper results.

The fixed experiment registry covers every main-table row, but blocks execution
when the manuscript does not determine an exact setting or historical artifact.
Known unresolved rows include frozen two-stage pretraining, SBERT-only selection,
RelGT variants, TNCN, and additional temporal fraud head settings. Supported rows
combine manuscript-stated values with explicitly labeled retained-code defaults;
they should not be described as recovered original run metadata.

Git objects, commit authors and messages, configured remotes, local backups and
the archived snapshots are outside the cleaned source snapshot. Historical files
can still be retrieved there. Use the source-only export when sharing this
cleanup; copying the entire working directory would include that history.

There is no project-wide license file in this checkout. Existing baseline license
files are retained, but this cleanup does not establish licensing or publication
permission for the whole project.
