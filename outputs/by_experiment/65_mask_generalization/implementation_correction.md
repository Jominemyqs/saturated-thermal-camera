# Extended-block ordering correction

During the first execution, an implementation error was found in the new
output-65 driver (not in the shared sampler or output 64). It constructed the
extended covariance with concatenated IDs `[baseline_ids, added_ids]`, whereas
`additional_layout` returns sorted union IDs. Thus the mask likelihood was
attached to incorrectly ordered means/covariance columns. Baseline C0 uses only
baseline IDs and was unaffected. Previous-state inference was also unaffected.

The initial development replay checked the baseline blocks but missed the
extended-block ordering. All initial mask runs are invalid computational
attempts: Diagonal 7 and Horizontal 12 reached their watchdog limit; Spiral 1
was interrupted when the bug was discovered. None completed or yielded a
saved scientific mask posterior. Their elapsed times must NOT be interpreted
as mask runtime or generalization evidence.

The original driver/helper/tests are preserved as `*_before_ordering_fix.py.txt`,
alongside the initial fingerprints and stopping decisions. New aligned prior
files have a distinct name; original caches are not overwritten. The correction
reuses the proven `extend_blocks` helper from output 64, records observation IDs,
and asserts their agreement with likelihood IDs before sampling. A regression
test explicitly fails on the old concatenation ordering. The expanded replay
checks every extended block and the actual development mask posterior/draws
against output 64 before resuming held-out fits.

Only ordering-affected mask attempts are restarted with unchanged seeds, chain
lengths, likelihood, priors and gates. Completed valid C0 fits are reused.
Original predecessor-stage stops for Diagonal 1 and Horizontal 1 are preserved,
not reopened under this correction. Horizontal 1's longer chains passed the
statistical screen, but deterministic prior construction exceeded the combined
900-second preparation-stage watchdog; that is not a statistical convergence
failure. No additional prior sampling is authorized for these stopped cases.

The prospective trajectory/ceiling selection, U sets, all scientific settings,
and higher-ceiling gate are unchanged. The 320 K posterior comparison remains
gated out if any selected reference case is incomplete. Corrected source hashes
are recorded separately in `ordering_fix_fingerprints.json`; initial hashes are
retained rather than silently replaced.
