#
# :date: 2018-08-10
# :author: PN
# :maintainer: PN
# :copyright: GPL v2 or later
#
# JailData/county/pierce/charges/src/Makefile
#

all: \
		output/pierce.csv.gz

.PHONY: all

output/pierce.csv.gz: \
		src/charges.py \
		input/pierce.csv.gz \
		hand/booking_charges.csv \
		hand/fugitive.csv \
		hand/seriousness.yaml
	python $< \
		--pierce=input/pierce.csv.gz \
		--booking_charges=hand/booking_charges.csv \
		--fugitive_charges=hand/fugitive.csv \
		--seriousness=hand/seriousness.yaml \
		--output=$@

# END.