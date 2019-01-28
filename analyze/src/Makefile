#
# :date: 2018-01-28
# :author: PN
# :maintainer: PN
# :copyright: GPL v2 or later
#
# ice-detainers-pierce/analyze/src/Makefile
#

all: \
		output/data.yaml

.phony: all

output/data.yaml: \
		src/analyze.py \
		input/pierce.csv.gz
	python $< \
		--pierce=input/pierce.csv.gz \
		--output=$@

# END.