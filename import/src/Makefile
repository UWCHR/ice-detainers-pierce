#
# :date: 2019-01-28
# :author: PN
# :maintainer: PN
# :copyright: GPL v2 or later
#
# ice-detainers-pierce/import/src/Makefile
#

all: \
		output/pierce.csv.gz \
		output/bail.csv.gz \
		output/birthplaces.csv.gz \
		output/setup-vars.yaml

.PHONY: all

output/pierce.csv.gz: \
		input/pierce.csv.gz
	cd output && ln -sf ../$< .

output/bail.csv.gz: \
		input/bail.csv.gz
	cd output && ln -sf ../$< .

output/birthplaces.csv.gz: \
		input/birthplaces.csv.gz
	cd output && ln -sf ../$< .

output/setup-vars.yaml: \
		input/setup-vars.yaml
	cd output && ln -sf ../$< .

# END.