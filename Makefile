.PHONY: all versions imports flowers clean
.SILENT:

all: versions imports flowers

versions:
	python 01_check_versions.py

imports:
	python 02_check_imports.py

flowers: data/iris.csv
	python 03_analyze_flowers.py

clean-windows:
	del /q "figures\*.png"

clean-binder:
	rm -r figures/*