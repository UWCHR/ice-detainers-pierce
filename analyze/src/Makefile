#
# :date: 2018-08-13
# :author: PN
# :maintainer: PN
# :copyright: GPL v2 or later
#
# JailData/county/pierce/analyze/src/Makefile
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