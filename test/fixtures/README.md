# Pipeline reference fixtures

`cases.json` contains synthetic extraction responses, manifest scenarios,
and RGBA pixels. These checks exercise pipeline behavior, not visual model
quality. `tracker-export.json` pins the tracker-generated 2.10.2 schema and
taxonomy independently of sibling checkouts.

`python-reference.json` is the frozen compatibility baseline captured before
the TypeScript cutover. It contains raw/debug/normalized/final rows, filter
reports, manifest selection, color results, completion cases, and an offline
refresh. Its historical filename records provenance; tests require only Node.
Timestamps and temporary paths are normalized; record and array order remain.

Offline tests compare prompts, normalization, overrides, manifests, refresh,
and colors against that baseline. Five RGBA cases cover transparency and
highlights, empty icons, hair face bias, ties-to-even rounding, and taupe hair
merging. JSON syntax-error wording is compared by stable error category.

Run `npm run check`. Do not regenerate expected output from the implementation
under test. Add independently reviewed expectations for intentional behavior
changes, preserving the unchanged baseline. Change the pinned taxonomy only
with an explicitly reviewed schema/fixture update.

`src/constants/extraction-rules.json` holds maintained prompt and alias rules
ported from the previous runtime; it is not a copy of the tracker taxonomy.
