# Pierce County ICE Detainer Data

Author: [@philneff](https://github.com/philneff) philneff@uw.edu

Date: 2019-01-28

License: GPL v2 or later

This project analyzes jail booking data regarding ICE and other federal holds, testing the hypothesis that ICE holds for local detainees lead to increased jail time.

This public repository reflects the state of this project at the time of publication of the January 30, 2019 version of the report ["Unequal Justice: Measuring the Impact of ICE Detainers on Jail Time in Pierce County"](https://jsis.washington.edu/humanrights/uwchr-pierce-county-ice-detainers-final/). This repository may be updated with bugfixes or corrections, but further analysis will proceed in a private repository maintained by [@philneff](https://github.com/philneff).

This project uses "Principled Data Processing" techniques and tools developed by [@HRDAG](https://github.com/HRDAG); see for example ["The Task Is A Quantum of Workflow."](https://hrdag.org/2016/06/14/the-task-is-a-quantum-of-workflow/)

Tasks in this project are designed to be executed using the recursive Make tool [makr](https://github.com/hrdag/makr).

## Task structure

`import/` - Convenience folder for conversion of initial input files into symbolic links in `import/output/`, which are then linked to input in downstream tasks. Input files have been previously processed from six separate public records releases by Pierce County to merge files for analysis and to drop personally indentifiable information. The field 'booking_id' is converted to a hash value which is used as a key to merge records in future steps.

`get_variables/` - Sets up dummy variables for analysis.

`charges/` - Merges in hand-coded type and seriousness values for booking charges based on the [Revised Code of Washington (RCW)](https://apps.leg.wa.gov/rcw/), and determines type and seriousness of the maximum charge for each booking. Bookings of type 'other' are dropped.

`bail/` - Merges in bail payment data and creates a dummy variable for whether any bail was posted for each booking.

`analyze/` - Primary statistical analysis script which outputs variables, tables, and figures.

`write/` - knitr script to output report as PDF using LaTex. Report variables are imported from CSV or YAML files output by upstream tasks; we do some light calculations here using R.

## Notes

We note that there are many improvements to make here in terms of code syntax and cleanliness, testing, and migration of setup and analysis tasks to functions; as well as implementing additional LaTeX processing of references and other aspects of the final report. We hope to improve on these aspects of this project for future publication.
