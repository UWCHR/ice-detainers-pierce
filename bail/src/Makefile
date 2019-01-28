#
# :date: 2018-01-28
# :author: PN
# :maintainer: PN
# :copyright: GPL v2 or later
#
# ice-detainers-pierce/bail/src/Makefile
#

all: \
		output/pierce.csv.gz

.phony: all

output/pierce.csv.gz: \
		src/bail.py \
		input/pierce.csv.gz \
		input/bail.csv.gz
	python $< \
		--pierce=input/pierce.csv.gz \
		--bail=input/bail.csv.gz \
		--output=$@

# END.