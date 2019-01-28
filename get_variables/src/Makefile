#
# :date: 2019-01-28
# :author: PN
# :maintainer: PN
# :copyright: GPL v2 or later
#
# ice-detainers-pierce/get_variables/src/Makefile
#

all: \
		output/pierce.csv.gz

.PHONY: all

output/pierce.csv.gz: \
		src/get_variables.py \
		input/pierce.csv.gz
	python $< \
		--pierce=input/pierce.csv.gz \
		--output=$@

# END.